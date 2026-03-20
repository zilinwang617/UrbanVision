import pandas as pd
import requests
import time

# read CSV
df = pd.read_csv("cleaned_data.csv", dtype={"parcel_id": str})
df["parcel_id"] = df["parcel_id"].str.strip()

# only test for 10 lines
df_test = df.head(10)

print("Total parcels to test:", len(df_test))
print("Start processing...\n")

tax_status_list = []

for i, parcel_id in enumerate(df_test["parcel_id"]):
    print(f"[{i+1}/{len(df_test)}] Checking: {parcel_id}")

    url = f"https://tools.wprdc.org/property-api/v0/parcels/{parcel_id}"

    try:
        response = requests.get(url, timeout=10,
                                headers={"User-Agent": "Mozilla/5.0"})
        
        if response.status_code != 200:
            print(f"  HTTP {response.status_code} for {parcel_id}")
            tax_status_list.append("NotFound")
            time.sleep(0.2)
            continue

        data = response.json()

        if not data.get("results"):
            tax_status = "NotFound"
        else:
            result = data["results"][0]
            datasets = result.get("data", {})

            tax_delinquency = datasets.get("pgh_tax_delinquency", [])
            tax_liens = datasets.get("tax_liens", [])

            if len(tax_delinquency) == 0 and len(tax_liens) == 0:
                tax_status = "Normal"
            else:
                tax_status = "Delinquent"

        tax_status_list.append(tax_status)

    except Exception as e:
        print(f"  Error with {parcel_id}: {e}")
        tax_status_list.append("Error")

    time.sleep(0.2)

df_test["tax_status"] = tax_status_list
df_test.to_csv("parcel_with_tax_status_test.csv", index=False)

print("\nFinished! Test CSV saved as 'parcel_with_tax_status_test.csv'")