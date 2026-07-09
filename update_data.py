"""Update fight data to include the latest results.

The Greco1899/scrape_ufc_stats repo re-scrapes UFCStats after every event,
so pulling its CSVs and re-running the adapter refreshes everything the
model trains on. Caches are cleared so features rebuild.

Usage: python update_data.py
"""

import glob
import os
import urllib.request

FILES = ["ufc_event_details.csv", "ufc_fight_results.csv",
         "ufc_fight_stats.csv", "ufc_fighter_tott.csv"]
BASE = "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main/"


def main():
    os.makedirs("raw", exist_ok=True)
    for f in FILES:
        print(f"downloading {f} ...")
        urllib.request.urlretrieve(BASE + f, f"raw/{f}")

    import adapter
    df = adapter.build("raw")
    df.to_csv("fights_v2.csv", index=False)
    print(f"fights_v2.csv rebuilt: {len(df)} fights "
          f"through {df['date'].max().date()}")

    for c in glob.glob("cache_*.pkl"):
        os.remove(c)
        print(f"cleared {c}")


if __name__ == "__main__":
    main()
