import csv
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


@dataclass
class PanoramaCandidate:
    pano_id: str
    lat: float
    lon: float
    heading: float
    pitch: float | None
    roll: float | None
    date: str | None
    elevation: float | None

def safe_entry_id(value: Any) -> str:
    """Make sure filenames are stable and filesystem-safe."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip())
    return cleaned.strip("_") or "entry"


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return heading in degrees from point1 (camera) to point2 (house)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    brng = math.degrees(math.atan2(y, x))
    return (brng + 360) % 360


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in meters between two lat/lng pairs."""
    radius_m = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * radius_m * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def format_lat_lng(lat: float, lon: float) -> str:
    """Format a lat/lng pair the way Google web services expect."""
    return f"{lat:.8f},{lon:.8f}"


def normalize_house_address(address: str | None, default_state: str = "PA") -> str | None:
    """Normalize CSV address strings so geocoding is more likely to hit the exact parcel."""
    address = (address or "").strip()
    if not address:
        return None
    if address.lower() == "no primary address specified":
        return None

    normalized = re.sub(r"\s+", " ", address)
    normalized = re.sub(r"\s+,", ",", normalized)

    if re.search(r",\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?$", normalized, re.IGNORECASE):
        return normalized

    zip_match = re.search(r",\s*(\d{5}(?:-\d{4})?)$", normalized)
    if zip_match:
        return re.sub(
            r",\s*(\d{5}(?:-\d{4})?)$",
            f", {default_state} \\1",
            normalized,
        )

    if not re.search(r",\s*[A-Z]{2}(?:\s+\d{5}(?:-\d{4})?)?$", normalized, re.IGNORECASE):
        return f"{normalized}, {default_state}"

    return normalized


def geocode_address(address: str, api_key: str, region: str = "us") -> dict[str, Any]:
    """Convert an address string into a geocoded point."""
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params: dict[str, Any] = {
        "address": address,
        "key": api_key,
    }
    if region:
        params["region"] = region

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def summarize_geocode_result(geocode_response: dict[str, Any]) -> dict[str, Any] | None:
    """Keep the most relevant geocode fields for downstream decision-making."""
    if geocode_response.get("status") != "OK" or not geocode_response.get("results"):
        return None

    result = geocode_response["results"][0]
    geometry = result.get("geometry", {})
    location = geometry.get("location", {})
    return {
        "formatted_address": result.get("formatted_address"),
        "place_id": result.get("place_id"),
        "partial_match": bool(result.get("partial_match")),
        "location_type": geometry.get("location_type"),
        "lat": location.get("lat"),
        "lon": location.get("lng"),
    }


def choose_house_target(
    address: str | None,
    api_key: str,
    fallback_lat: float | None = None,
    fallback_lon: float | None = None,
    region: str = "us",
    default_state: str = "PA",
) -> dict[str, Any]:
    """
    Prefer the street address whenever one exists.

    Principle:
    - if the CSV has a usable street address, use that path end-to-end;
    - only fall back to dataset coordinates when the address is missing or says
      'No primary address specified'.
    """
    normalized_address = normalize_house_address(address, default_state=default_state)
    geocode_response = None
    geocode_result = None
    target_lat = None
    target_lon = None
    target_source = None

    if normalized_address:
        geocode_response = geocode_address(normalized_address, api_key, region=region)
        geocode_result = summarize_geocode_result(geocode_response)

        if geocode_result and geocode_result.get("lat") is not None and geocode_result.get("lon") is not None:
            target_lat = geocode_result["lat"]
            target_lon = geocode_result["lon"]
            target_source = "geocode_address"

    if (target_lat is None or target_lon is None) and not normalized_address:
        if fallback_lat is not None and fallback_lon is not None:
            target_lat = fallback_lat
            target_lon = fallback_lon
            target_source = "dataset_coordinates"

    return {
        "normalized_address": normalized_address,
        "geocode_status": geocode_response.get("status") if geocode_response else None,
        "geocode_result": geocode_result,
        "geocode_response": geocode_response,
        "target_lat": target_lat,
        "target_lon": target_lon,
        "target_source": target_source,
    }


def _parse_capture_date(value: str | None) -> dict[str, Any] | None:
    value = (value or "").strip()
    if not value:
        return None

    for fmt, precision in (("%Y-%m-%d", "day"), ("%Y-%m", "month"), ("%Y", "year")):
        try:
            parsed = datetime.strptime(value, fmt)
            return {
                "value": value,
                "precision": precision,
                "year": parsed.year,
                "month": parsed.month if precision in {"day", "month"} else None,
            }
        except ValueError:
            continue

    return None


def _temporal_gap_months(target_capture_date: str | None, pano_date: str | None) -> int | None:
    target_info = _parse_capture_date(target_capture_date)
    pano_info = _parse_capture_date(pano_date)
    if not target_info or not pano_info:
        return None

    if target_info["precision"] == "year" or pano_info["precision"] == "year":
        return abs(target_info["year"] - pano_info["year"]) * 12

    return abs(
        (target_info["year"] - pano_info["year"]) * 12
        + (target_info["month"] - pano_info["month"])
    )


def make_search_url(lat: float, lon: float, radius: int = 50) -> str:
    """Build the URL of the script on Google's servers that returns nearby panoramas."""
    url = (
        "https://maps.googleapis.com/maps/api/js/"
        "GeoPhotoService.SingleImageSearch"
        "?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{0:}!4d{1:}!2d{2:d}!3m10"
        "!2m2!1sen!2sGB!9m1!1e2!11m4!1m3!1e2!2b1!3e2!4m10!1e1!1e2!1e3!1e4"
        "!1e8!1e6!5m1!1e2!6m1!1e2"
        "&callback=callbackfunc"
    )
    return url.format(lat, lon, radius)


def search_request(lat: float, lon: float, radius: int = 50, session: requests.Session | None = None) -> requests.Response:
    """Get the raw response that contains nearby panorama candidates."""
    client = session or requests
    return client.get(make_search_url(lat, lon, radius=radius), timeout=30)


def extract_panoramas(text: str) -> list[PanoramaCandidate]:
    """Parse panorama candidates from Google's callback payload."""
    match = re.search(r"callbackfunc\(\s*(.*)\s*\)$", text.strip())
    if not match:
        raise ValueError("Unexpected Street View candidate payload.")

    data = json.loads(match.group(1))
    if data == [[5, "generic", "Search returned no images."]]:
        return []

    try:
        subset = data[1][5][0]
        raw_panos = subset[3][0]
        raw_dates = [] if (len(subset) < 9 or subset[8] is None) else subset[8]
    except (IndexError, TypeError) as exc:
        raise ValueError("Street View candidate payload format changed.") from exc

    raw_panos = raw_panos[::-1]
    raw_dates = raw_dates[::-1]
    dates = [f"{d[1][0]}-{d[1][1]:02d}" for d in raw_dates]

    candidates: list[PanoramaCandidate] = []
    for index, pano in enumerate(raw_panos):
        candidates.append(
            PanoramaCandidate(
                pano_id=pano[0][1],
                lat=pano[2][0][2],
                lon=pano[2][0][3],
                heading=pano[2][2][0],
                pitch=pano[2][2][1] if len(pano[2][2]) >= 2 else None,
                roll=pano[2][2][2] if len(pano[2][2]) >= 3 else None,
                date=dates[index] if index < len(dates) else None,
                elevation=pano[3][0] if len(pano) >= 4 else None,
            )
        )

    return candidates


def search_panoramas(lat: float, lon: float, radius: int = 50, session: requests.Session | None = None) -> list[PanoramaCandidate]:
    """Get nearby panorama candidates for a coordinate."""
    response = search_request(lat, lon, radius=radius, session=session)
    response.raise_for_status()
    return extract_panoramas(response.text)


def get_panorama_metadata(
    pano_id: str,
    api_key: str,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Return metadata for a specific panorama id."""
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    client = session or requests
    response = client.get(url, params={"pano": pano_id, "key": api_key}, timeout=30)
    response.raise_for_status()
    return response.json()


def get_streetview_metadata(
    location: str | tuple[float, float],
    api_key: str,
    radius: int | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Return Street View metadata for an address string or a lat/lng pair."""
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    client = session or requests
    params: dict[str, Any] = {"key": api_key}
    if isinstance(location, str):
        params["location"] = location
    else:
        params["location"] = format_lat_lng(location[0], location[1])
        if radius is not None:
            params["radius"] = radius

    response = client.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def get_reference_pano(
    normalized_address: str | None,
    target_lat: float,
    target_lon: float,
    api_key: str,
    radius: int = 20,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """
    Resolve the baseline pano that defines the trusted camera position.

    Preference order:
    - use direct Street View metadata on the normalized address when available;
    - otherwise use Street View metadata on the chosen target coordinates.
    """
    if normalized_address:
        baseline_source = "address_metadata"
        metadata = get_streetview_metadata(normalized_address, api_key, session=session)
    else:
        baseline_source = "coordinate_metadata"
        metadata = get_streetview_metadata((target_lat, target_lon), api_key, radius=radius, session=session)

    if metadata.get("status") != "OK":
        return {
            "ok": False,
            "status": metadata.get("status") or "UNKNOWN_ERROR",
            "message": metadata.get("error_message"),
            "baseline_source": baseline_source,
            "metadata": metadata,
        }

    location = metadata.get("location") or {}
    cam_lat = location.get("lat")
    cam_lon = location.get("lng")
    if cam_lat is None or cam_lon is None:
        return {
            "ok": False,
            "status": "BASELINE_LOCATION_MISSING",
            "message": "Street View baseline metadata did not contain a camera location.",
            "baseline_source": baseline_source,
            "metadata": metadata,
        }

    return {
        "ok": True,
        "status": "OK",
        "baseline_source": baseline_source,
        "pano_id": metadata.get("pano_id"),
        "pano_date": metadata.get("date"),
        "cam_lat": cam_lat,
        "cam_lon": cam_lon,
        "distance_from_target_meters": haversine_meters(target_lat, target_lon, cam_lat, cam_lon),
        "metadata": metadata,
    }


def _dedupe_candidates(candidates: list[PanoramaCandidate]) -> list[PanoramaCandidate]:
    seen: set[str] = set()
    deduped: list[PanoramaCandidate] = []
    for candidate in candidates:
        if candidate.pano_id in seen:
            continue
        seen.add(candidate.pano_id)
        deduped.append(candidate)
    return deduped


def _reduce_to_nearest_per_timepoint(candidate_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Keep only the nearest pano for each capture date.

    Principle:
    - if multiple pano ids belong to the same Street View timepoint, the one closest
      to the baseline pano position is the best proxy for that timepoint;
    - undated candidates are each kept, because they cannot be safely grouped.
    """
    nearest_by_date: dict[str, dict[str, Any]] = {}
    undated_candidates: list[dict[str, Any]] = []

    for candidate in candidate_records:
        pano_date = candidate.get("pano_date")
        if not pano_date:
            undated_candidates.append(candidate)
            continue

        current = nearest_by_date.get(pano_date)
        candidate_key = (
            float("inf") if candidate.get("baseline_distance_meters") is None else candidate["baseline_distance_meters"],
            candidate["distance_meters"],
            candidate["pano_id"],
        )
        current_key = (
            float("inf") if current.get("baseline_distance_meters") is None else current["baseline_distance_meters"],
            current["distance_meters"],
            current["pano_id"],
        ) if current else None

        if current is None or candidate_key < current_key:
            nearest_by_date[pano_date] = candidate

    return list(nearest_by_date.values()) + undated_candidates


def _build_candidate_record(
    candidate: PanoramaCandidate,
    target_lat: float,
    target_lon: float,
    target_capture_date: str | None,
    baseline_lat: float | None,
    baseline_lon: float | None,
    api_key: str,
    session: requests.Session,
) -> dict[str, Any]:
    cam_lat = candidate.lat
    cam_lon = candidate.lon
    pano_date = candidate.date
    date_source = "search_candidate" if pano_date else None
    metadata_status = None
    metadata_error = None

    if not pano_date:
        try:
            metadata = get_panorama_metadata(candidate.pano_id, api_key, session=session)
            metadata_status = metadata.get("status")
            if metadata_status == "OK":
                location = metadata.get("location") or {}
                if location.get("lat") is not None and location.get("lng") is not None:
                    cam_lat = location["lat"]
                    cam_lon = location["lng"]
                if metadata.get("date"):
                    pano_date = metadata["date"]
                    date_source = "metadata_lookup"
        except requests.RequestException as exc:
            metadata_error = str(exc)

    temporal_gap_months = _temporal_gap_months(target_capture_date, pano_date)
    baseline_distance_meters = None
    if baseline_lat is not None and baseline_lon is not None:
        baseline_distance_meters = haversine_meters(baseline_lat, baseline_lon, cam_lat, cam_lon)

    return {
        "pano_id": candidate.pano_id,
        "cam_lat": cam_lat,
        "cam_lon": cam_lon,
        "distance_meters": haversine_meters(target_lat, target_lon, cam_lat, cam_lon),
        "baseline_distance_meters": baseline_distance_meters,
        "pano_date": pano_date,
        "date_source": date_source,
        "temporal_gap_months": temporal_gap_months,
        "temporal_gap_years": round(temporal_gap_months / 12, 3) if temporal_gap_months is not None else None,
        "heading_hint": candidate.heading,
        "pitch_hint": candidate.pitch,
        "roll_hint": candidate.roll,
        "elevation": candidate.elevation,
        "metadata_status": metadata_status,
        "metadata_error": metadata_error,
    }


def _candidate_priority_key(candidate: dict[str, Any]) -> tuple[float, float, str]:
    return (
        float("inf") if candidate["baseline_distance_meters"] is None else candidate["baseline_distance_meters"],
        candidate["distance_meters"],
        candidate["pano_id"],
    )


def find_best_pano(
    target_lat: float,
    target_lon: float,
    api_key: str,
    target_capture_date: str | None = None,
    baseline_lat: float | None = None,
    baseline_lon: float | None = None,
    candidate_radius: int = 25,
    max_distance_meters: float = 20,
    max_baseline_shift_meters: float = 10,
    require_baseline: bool = True,
    allow_distance_fallback: bool = False,
) -> dict[str, Any]:
    """
    Search nearby panorama candidates, then choose the best pano by time and distance.

    Selection rule:
    - keep the address-driven/geocoded target point as the source of truth;
    - search a set of nearby pano candidates around that point;
    - first enforce a hard distance cap unless fallback is explicitly enabled;
    - require historical pano positions to stay close to the trusted baseline pano;
    - for each capture date, keep only the nearest pano around the baseline position;
    - inside that reduced pool, prefer the pano closest in time to the row's capture date;
    - break ties by distance.
    """
    if require_baseline and (baseline_lat is None or baseline_lon is None):
        return {
            "ok": False,
            "status": "NO_BASELINE_PANO",
            "message": "A baseline pano is required before historical pano selection can be constrained.",
            "candidate_count_total": 0,
            "candidate_count_considered": 0,
            "candidates": [],
        }

    with requests.Session() as session:
        try:
            raw_candidates = search_panoramas(target_lat, target_lon, radius=candidate_radius, session=session)
        except requests.RequestException as exc:
            return {
                "ok": False,
                "status": "REQUEST_ERROR",
                "message": str(exc),
                "candidate_count_total": 0,
                "candidate_count_considered": 0,
                "candidates": [],
            }
        except ValueError as exc:
            return {
                "ok": False,
                "status": "PARSE_ERROR",
                "message": str(exc),
                "candidate_count_total": 0,
                "candidate_count_considered": 0,
                "candidates": [],
            }

        unique_candidates = _dedupe_candidates(raw_candidates)
        if not unique_candidates:
            return {
                "ok": False,
                "status": "ZERO_RESULTS",
                "message": "No Street View panorama candidates found near the target coordinates.",
                "candidate_count_total": 0,
                "candidate_count_considered": 0,
                "candidates": [],
            }

        candidate_records = [
            _build_candidate_record(
                candidate=candidate,
                target_lat=target_lat,
                target_lon=target_lon,
                target_capture_date=target_capture_date,
                baseline_lat=baseline_lat,
                baseline_lon=baseline_lon,
                api_key=api_key,
                session=session,
            )
            for candidate in unique_candidates
        ]

        target_guarded_pool = [
            candidate
            for candidate in candidate_records
            if max_distance_meters is None or candidate["distance_meters"] <= max_distance_meters
        ]
        if max_distance_meters is not None and not target_guarded_pool and not allow_distance_fallback:
            return {
                "ok": False,
                "status": "NO_CLOSE_PANO",
                "message": (
                    f"No Street View panorama candidate was found within {max_distance_meters} meters "
                    "of the target point."
                ),
                "candidate_count_total": len(candidate_records),
                "candidate_count_considered": 0,
                "candidate_count_target_filtered": 0,
                "candidate_count_baseline_filtered": 0,
                "distance_guardrail_applied": True,
                "distance_guardrail_meters": max_distance_meters,
                "candidate_radius_meters": candidate_radius,
                "max_baseline_shift_meters": max_baseline_shift_meters,
                "require_baseline": require_baseline,
                "allow_distance_fallback": allow_distance_fallback,
                "candidates": candidate_records,
            }

        target_considered_candidates = target_guarded_pool if target_guarded_pool else candidate_records
        baseline_guarded_pool = [
            candidate
            for candidate in target_considered_candidates
            if max_baseline_shift_meters is None
            or candidate["baseline_distance_meters"] is None
            or candidate["baseline_distance_meters"] <= max_baseline_shift_meters
        ]

        if (
            baseline_lat is not None
            and baseline_lon is not None
            and max_baseline_shift_meters is not None
            and not baseline_guarded_pool
            and not allow_distance_fallback
        ):
            return {
                "ok": False,
                "status": "NO_BASELINE_ALIGNED_PANO",
                "message": (
                    f"No Street View panorama candidate was found within {max_baseline_shift_meters} meters "
                    "of the baseline pano position."
                ),
                "candidate_count_total": len(candidate_records),
                "candidate_count_considered": 0,
                "candidate_count_target_filtered": len(target_considered_candidates),
                "candidate_count_baseline_filtered": 0,
                "distance_guardrail_applied": bool(target_guarded_pool),
                "distance_guardrail_meters": max_distance_meters,
                "candidate_radius_meters": candidate_radius,
                "max_baseline_shift_meters": max_baseline_shift_meters,
                "require_baseline": require_baseline,
                "allow_distance_fallback": allow_distance_fallback,
                "candidates": candidate_records,
            }

        considered_candidates = baseline_guarded_pool if baseline_guarded_pool else target_considered_candidates
        reduced_candidates = _reduce_to_nearest_per_timepoint(considered_candidates)

        if target_capture_date:
            dated_candidates = [
                candidate
                for candidate in reduced_candidates
                if candidate["temporal_gap_months"] is not None
            ]
            if dated_candidates:
                best_candidate = min(
                    dated_candidates,
                    key=lambda item: (
                        item["temporal_gap_months"],
                        *_candidate_priority_key(item),
                    ),
                )
                selection_reason = "min_temporal_gap_then_baseline_distance_then_distance"
            else:
                best_candidate = min(
                    reduced_candidates,
                    key=_candidate_priority_key,
                )
                selection_reason = "no_dated_candidates_pick_nearest_to_baseline"
        else:
            best_candidate = min(
                reduced_candidates,
                key=_candidate_priority_key,
            )
            selection_reason = "no_target_capture_date_pick_nearest_to_baseline"

        selected_metadata = None
        selected_metadata_error = None
        try:
            selected_metadata = get_panorama_metadata(best_candidate["pano_id"], api_key, session=session)
            if selected_metadata.get("status") == "OK":
                location = selected_metadata.get("location") or {}
                if location.get("lat") is not None and location.get("lng") is not None:
                    best_candidate["cam_lat"] = location["lat"]
                    best_candidate["cam_lon"] = location["lng"]
                    best_candidate["distance_meters"] = haversine_meters(
                        target_lat,
                        target_lon,
                        best_candidate["cam_lat"],
                        best_candidate["cam_lon"],
                    )
                    if baseline_lat is not None and baseline_lon is not None:
                        best_candidate["baseline_distance_meters"] = haversine_meters(
                            baseline_lat,
                            baseline_lon,
                            best_candidate["cam_lat"],
                            best_candidate["cam_lon"],
                        )
                if selected_metadata.get("date"):
                    best_candidate["pano_date"] = selected_metadata["date"]
                    best_candidate["date_source"] = "metadata_lookup"
                    best_candidate["temporal_gap_months"] = _temporal_gap_months(
                        target_capture_date,
                        best_candidate["pano_date"],
                    )
                    temporal_gap_months = best_candidate["temporal_gap_months"]
                    best_candidate["temporal_gap_years"] = (
                        round(temporal_gap_months / 12, 3) if temporal_gap_months is not None else None
                    )
        except requests.RequestException as exc:
            selected_metadata_error = str(exc)

        return {
            "ok": True,
            "status": "OK",
            "selection_reason": selection_reason,
            "candidate_count_total": len(candidate_records),
            "candidate_count_considered": len(considered_candidates),
            "candidate_count_target_filtered": len(target_considered_candidates),
            "candidate_count_baseline_filtered": len(baseline_guarded_pool),
            "candidate_count_timepoints": len(reduced_candidates),
            "distance_guardrail_applied": bool(target_guarded_pool),
            "distance_guardrail_meters": max_distance_meters,
            "candidate_radius_meters": candidate_radius,
            "max_baseline_shift_meters": max_baseline_shift_meters,
            "require_baseline": require_baseline,
            "allow_distance_fallback": allow_distance_fallback,
            "target_capture_date": target_capture_date,
            "baseline_lat": baseline_lat,
            "baseline_lon": baseline_lon,
            "best_candidate": best_candidate,
            "selected_metadata": selected_metadata,
            "selected_metadata_error": selected_metadata_error,
            "candidates": candidate_records,
            "timepoint_candidates": reduced_candidates,
        }


def download_sv_image(
    pano_id: str | None,
    cam_lat: float,
    cam_lon: float,
    heading: float,
    api_key: str,
    out_path: str | Path,
    size: str = "640x640",
    fov: int = 80,
    pitch: int = 0,
    return_error_code: bool = True,
) -> str:
    """Download a Street View image for a specific pano so the image request does not snap again."""
    url = "https://maps.googleapis.com/maps/api/streetview"
    params: dict[str, Any] = {
        "size": size,
        "fov": fov,
        "pitch": pitch,
        "heading": heading,
        "key": api_key,
    }
    if return_error_code:
        params["return_error_code"] = "true"
    if pano_id:
        params["pano"] = pano_id
    else:
        params["location"] = format_lat_lng(cam_lat, cam_lon)

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(response.content)
    return str(out_path)


def write_sv_metadata(meta_path: str | Path, payload: dict[str, Any]) -> str:
    """Save Street View metadata next to the downloaded image."""
    meta_path = Path(meta_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(meta_path)


def fetch_front_gsv_for_house(
    address: str | None,
    api_key: str,
    house_lat: float | None = None,
    house_lon: float | None = None,
    target_capture_date: str | None = None,
    out_dir: str | Path = "gsv_out",
    entry_id: str = "sample",
    candidate_radius: int = 25,
    max_distance_meters: float = 20,
    max_baseline_shift_meters: float = 10,
    require_baseline: bool = True,
    allow_distance_fallback: bool = False,
    size: str = "640x640",
    fov: int = 80,
    pitch: int = 0,
    region: str = "us",
    default_state: str = "PA",
) -> dict[str, Any]:
    """
    Fetch a front-facing Street View image using address-first geocoding and coordinate-based pano selection.

    The geocoded coordinate remains the source of truth. Street View lookup is then
    done against nearby pano candidates around that point, instead of directly asking
    Google for the single nearest pano by address string.
    """
    target = choose_house_target(
        address=address,
        api_key=api_key,
        fallback_lat=house_lat,
        fallback_lon=house_lon,
        region=region,
        default_state=default_state,
    )

    target_lat = target["target_lat"]
    target_lon = target["target_lon"]
    if target_lat is None or target_lon is None:
        return {
            "ok": False,
            "status": "INVALID_REQUEST",
            "message": "Missing both a geocodable address and fallback coordinates.",
            "target": target,
        }

    baseline = get_reference_pano(
        normalized_address=target["normalized_address"],
        target_lat=target_lat,
        target_lon=target_lon,
        api_key=api_key,
        radius=candidate_radius,
    )
    if require_baseline and not baseline.get("ok"):
        return {
            "ok": False,
            "status": baseline.get("status"),
            "message": baseline.get("message"),
            "lookup_source": "candidate_search_coordinates",
            "baseline": baseline,
            "target": target,
        }

    selection = find_best_pano(
        target_lat=target_lat,
        target_lon=target_lon,
        api_key=api_key,
        target_capture_date=target_capture_date,
        baseline_lat=baseline.get("cam_lat"),
        baseline_lon=baseline.get("cam_lon"),
        candidate_radius=candidate_radius,
        max_distance_meters=max_distance_meters,
        max_baseline_shift_meters=max_baseline_shift_meters,
        require_baseline=require_baseline,
        allow_distance_fallback=allow_distance_fallback,
    )

    if not selection.get("ok"):
        return {
            "ok": False,
            "status": selection.get("status"),
            "message": selection.get("message"),
            "lookup_source": "candidate_search_coordinates",
            "lookup_attempts": [
                {
                    "lookup_type": "candidate_search",
                    "lookup_value": format_lat_lng(target_lat, target_lon),
                    "radius": candidate_radius,
                    "distance_guardrail_meters": max_distance_meters,
                    "baseline_shift_guardrail_meters": max_baseline_shift_meters,
                    "require_baseline": require_baseline,
                    "allow_distance_fallback": allow_distance_fallback,
                }
            ],
            "baseline": baseline,
            "target": target,
            "selection": selection,
        }

    best_candidate = selection["best_candidate"]
    cam_lat = best_candidate["cam_lat"]
    cam_lon = best_candidate["cam_lon"]
    heading = bearing_deg(cam_lat, cam_lon, target_lat, target_lon)
    distance_meters = best_candidate["distance_meters"]

    out_path = Path(out_dir) / f"{entry_id}_front.jpg"
    saved_path = download_sv_image(
        pano_id=best_candidate["pano_id"],
        cam_lat=cam_lat,
        cam_lon=cam_lon,
        heading=heading,
        api_key=api_key,
        out_path=out_path,
        size=size,
        fov=fov,
        pitch=pitch,
    )

    selected_metadata = selection.get("selected_metadata") or {
        "status": "OK",
        "pano_id": best_candidate["pano_id"],
        "date": best_candidate["pano_date"],
        "location": {
            "lat": cam_lat,
            "lng": cam_lon,
        },
    }

    candidate_summaries = []
    for candidate in selection["candidates"]:
        candidate_summaries.append(
            {
                "pano_id": candidate["pano_id"],
                "cam_lat": candidate["cam_lat"],
                "cam_lon": candidate["cam_lon"],
                "distance_meters": candidate["distance_meters"],
                "baseline_distance_meters": candidate["baseline_distance_meters"],
                "pano_date": candidate["pano_date"],
                "date_source": candidate["date_source"],
                "temporal_gap_months": candidate["temporal_gap_months"],
                "temporal_gap_years": candidate["temporal_gap_years"],
                "selected": candidate["pano_id"] == best_candidate["pano_id"],
            }
        )

    review_summary = {
        "selected_pano_id": best_candidate["pano_id"],
        "selected_pano_date": best_candidate["pano_date"],
        "target_capture_date": target_capture_date,
        "temporal_gap_months": best_candidate["temporal_gap_months"],
        "temporal_gap_years": best_candidate["temporal_gap_years"],
        "selected_distance_to_target_meters": distance_meters,
        "selected_distance_to_baseline_meters": best_candidate["baseline_distance_meters"],
        "baseline_pano_id": baseline.get("pano_id"),
        "baseline_pano_date": baseline.get("pano_date"),
        "baseline_distance_from_target_meters": baseline.get("distance_from_target_meters"),
        "candidate_radius_meters": candidate_radius,
        "distance_guardrail_meters": max_distance_meters,
        "baseline_shift_guardrail_meters": max_baseline_shift_meters,
        "within_target_distance_guardrail": (
            max_distance_meters is None or distance_meters <= max_distance_meters
        ),
        "within_baseline_shift_guardrail": (
            max_baseline_shift_meters is None
            or best_candidate["baseline_distance_meters"] is None
            or best_candidate["baseline_distance_meters"] <= max_baseline_shift_meters
        ),
        "selection_reason": selection["selection_reason"],
        "candidate_count_total": selection["candidate_count_total"],
        "candidate_count_target_filtered": selection["candidate_count_target_filtered"],
        "candidate_count_baseline_filtered": selection["candidate_count_baseline_filtered"],
        "candidate_count_timepoints": selection["candidate_count_timepoints"],
        "heading_degrees": heading,
    }

    meta_path = out_path.with_suffix(".meta.json")
    saved_meta_path = write_sv_metadata(
        meta_path,
        {
            "review_summary": review_summary,
            "entry_id": entry_id,
            "input_address": address,
            "normalized_address": target["normalized_address"],
            "fallback_house_lat": house_lat,
            "fallback_house_lon": house_lon,
            "target_lat": target_lat,
            "target_lon": target_lon,
            "target_source": target["target_source"],
            "target_capture_date": target_capture_date,
            "lookup_source": "candidate_search_coordinates",
            "baseline_source": baseline.get("baseline_source"),
            "baseline_pano_id": baseline.get("pano_id"),
            "baseline_pano_date": baseline.get("pano_date"),
            "baseline_cam_lat": baseline.get("cam_lat"),
            "baseline_cam_lon": baseline.get("cam_lon"),
            "baseline_distance_from_target_meters": baseline.get("distance_from_target_meters"),
            "lookup_attempts": [
                {
                    "lookup_type": "candidate_search",
                    "lookup_value": format_lat_lng(target_lat, target_lon),
                    "radius": candidate_radius,
                    "distance_guardrail_meters": max_distance_meters,
                    "baseline_shift_guardrail_meters": max_baseline_shift_meters,
                    "require_baseline": require_baseline,
                    "allow_distance_fallback": allow_distance_fallback,
                    "candidate_count_total": selection["candidate_count_total"],
                    "candidate_count_considered": selection["candidate_count_considered"],
                    "candidate_count_target_filtered": selection["candidate_count_target_filtered"],
                    "candidate_count_baseline_filtered": selection["candidate_count_baseline_filtered"],
                    "candidate_count_timepoints": selection["candidate_count_timepoints"],
                    "selection_reason": selection["selection_reason"],
                }
            ],
            "cam_lat": cam_lat,
            "cam_lon": cam_lon,
            "heading": heading,
            "distance_meters": distance_meters,
            "baseline_distance_meters": best_candidate["baseline_distance_meters"],
            "temporal_gap_months": best_candidate["temporal_gap_months"],
            "temporal_gap_years": best_candidate["temporal_gap_years"],
            "candidate_count_total": selection["candidate_count_total"],
            "candidate_count_considered": selection["candidate_count_considered"],
            "candidate_count_target_filtered": selection["candidate_count_target_filtered"],
            "candidate_count_baseline_filtered": selection["candidate_count_baseline_filtered"],
            "candidate_count_timepoints": selection["candidate_count_timepoints"],
            "selection_reason": selection["selection_reason"],
            "image_path": saved_path,
            "geocode_status": target["geocode_status"],
            "geocode_result": target["geocode_result"],
            "geocode_response": target["geocode_response"],
            "gsv_metadata": selected_metadata,
            "candidate_search": {
                "candidate_radius_meters": candidate_radius,
                "distance_guardrail_meters": max_distance_meters,
                "baseline_shift_guardrail_meters": max_baseline_shift_meters,
                "require_baseline": require_baseline,
                "allow_distance_fallback": allow_distance_fallback,
                "selected_metadata_error": selection.get("selected_metadata_error"),
                "candidates": candidate_summaries,
            },
            "baseline_metadata": baseline.get("metadata"),
        },
    )

    return {
        "ok": True,
        "status": "OK",
        "review_summary": review_summary,
        "entry_id": entry_id,
        "input_address": address,
        "normalized_address": target["normalized_address"],
        "house_lat": target_lat,
        "house_lon": target_lon,
        "target_source": target["target_source"],
        "target_capture_date": target_capture_date,
        "lookup_source": "candidate_search_coordinates",
        "baseline_source": baseline.get("baseline_source"),
        "baseline_pano_id": baseline.get("pano_id"),
        "baseline_pano_date": baseline.get("pano_date"),
        "baseline_cam_lat": baseline.get("cam_lat"),
        "baseline_cam_lon": baseline.get("cam_lon"),
        "baseline_distance_from_target_meters": baseline.get("distance_from_target_meters"),
        "lookup_attempts": [
            {
                "lookup_type": "candidate_search",
                "lookup_value": format_lat_lng(target_lat, target_lon),
                "radius": candidate_radius,
                "distance_guardrail_meters": max_distance_meters,
                "baseline_shift_guardrail_meters": max_baseline_shift_meters,
                "require_baseline": require_baseline,
                "allow_distance_fallback": allow_distance_fallback,
                "candidate_count_total": selection["candidate_count_total"],
                "candidate_count_considered": selection["candidate_count_considered"],
                "candidate_count_target_filtered": selection["candidate_count_target_filtered"],
                "candidate_count_baseline_filtered": selection["candidate_count_baseline_filtered"],
                "candidate_count_timepoints": selection["candidate_count_timepoints"],
                "selection_reason": selection["selection_reason"],
            }
        ],
        "cam_lat": cam_lat,
        "cam_lon": cam_lon,
        "distance_meters": distance_meters,
        "baseline_distance_meters": best_candidate["baseline_distance_meters"],
        "heading": heading,
        "pano_id": best_candidate["pano_id"],
        "date": best_candidate["pano_date"],
        "temporal_gap_months": best_candidate["temporal_gap_months"],
        "temporal_gap_years": best_candidate["temporal_gap_years"],
        "candidate_count_total": selection["candidate_count_total"],
        "candidate_count_considered": selection["candidate_count_considered"],
        "candidate_count_target_filtered": selection["candidate_count_target_filtered"],
        "candidate_count_baseline_filtered": selection["candidate_count_baseline_filtered"],
        "candidate_count_timepoints": selection["candidate_count_timepoints"],
        "selection_reason": selection["selection_reason"],
        "image_path": saved_path,
        "meta_path": saved_meta_path,
        "baseline": baseline,
        "target": target,
        "meta": selected_metadata,
        "selection": selection,
    }


def load_house_entries(
    csv_path: str | Path,
    start: int = 51,
    end: int = 100,
    target_date_field: str = "create_date",
) -> list[dict[str, Any]]:
    """Load rows in the inclusive range [start, end] from cleaned_data.csv."""
    if start < 1:
        raise ValueError("start must be >= 1")
    if end < start:
        raise ValueError("end must be >= start")

    entries = []
    csv_path = Path(csv_path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            if index < start:
                continue
            if index > end:
                break

            address = (row.get("address") or "").strip()
            usable_address = bool(address) and address.lower() != "no primary address specified"
            lat_raw = (row.get("latitude") or "").strip()
            lon_raw = (row.get("longitude") or "").strip()
            has_coords = bool(lat_raw and lon_raw)

            if not usable_address and not has_coords:
                entries.append(
                    {
                        "index": index,
                        "row": row,
                        "skip_reason": "missing_address_and_coordinates",
                    }
                )
                continue

            entry = {
                "index": index,
                "row": row,
                "address": address or None,
                "entry_id": f"{index:03d}_{safe_entry_id(row.get('_id') or row.get('parcel_id') or 'entry')}",
                "target_capture_date": (row.get(target_date_field) or "").strip() or None,
            }
            if has_coords:
                entry["lat"] = float(lat_raw)
                entry["lon"] = float(lon_raw)

            entries.append(entry)

    return entries


def fetch_entries(
    csv_path: str | Path,
    api_key: str,
    start: int = 1,
    end: int = 50,
    out_dir: str | Path = "gsv_out",
    pause_seconds: float = 0.1,
    region: str = "us",
    default_state: str = "PA",
    target_date_field: str = "create_date",
    candidate_radius: int = 25,
    max_distance_meters: float = 20,
    max_baseline_shift_meters: float = 10,
    require_baseline: bool = True,
    allow_distance_fallback: bool = False,
) -> list[dict[str, Any]]:
    """Download front-facing Street View images for cleaned-data rows in [start, end]."""
    results = []
    for entry in load_house_entries(
        csv_path,
        start=start,
        end=end,
        target_date_field=target_date_field,
    ):
        row = entry["row"]
        result = {
            "index": entry["index"],
            "_id": row.get("_id"),
            "parcel_id": row.get("parcel_id"),
            "address": row.get("address"),
            "target_capture_date": entry.get("target_capture_date"),
        }

        if entry.get("skip_reason"):
            result.update(
                {
                    "ok": False,
                    "status": "SKIPPED",
                    "message": entry["skip_reason"],
                }
            )
            results.append(result)
            continue

        try:
            fetch_result = fetch_front_gsv_for_house(
                address=entry.get("address"),
                house_lat=entry.get("lat"),
                house_lon=entry.get("lon"),
                target_capture_date=entry.get("target_capture_date"),
                api_key=api_key,
                out_dir=out_dir,
                entry_id=entry["entry_id"],
                candidate_radius=candidate_radius,
                max_distance_meters=max_distance_meters,
                max_baseline_shift_meters=max_baseline_shift_meters,
                require_baseline=require_baseline,
                allow_distance_fallback=allow_distance_fallback,
                region=region,
                default_state=default_state,
            )
            result.update(fetch_result)
        except requests.HTTPError as exc:
            result.update(
                {
                    "ok": False,
                    "status": "HTTP_ERROR",
                    "message": str(exc),
                }
            )
        except requests.RequestException as exc:
            result.update(
                {
                    "ok": False,
                    "status": "REQUEST_ERROR",
                    "message": str(exc),
                }
            )
        except Exception as exc:
            result.update(
                {
                    "ok": False,
                    "status": type(exc).__name__,
                    "message": str(exc),
                }
            )

        results.append(result)
        if pause_seconds:
            time.sleep(pause_seconds)

    return results
