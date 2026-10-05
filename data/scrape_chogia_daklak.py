

import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

URL = "https://chogia.vn/gia-ca-phe-dak-lak-hom-nay/"
OUTPUT = r"C:/Users/ADMIN/OneDrive/Desktop/DAP391m/ketqua/ketqua.csv"   # doi duong dan neu can

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
}


def fetch_html(url, retries=3, delay=3):
    """Tai HTML, thu lai neu loi mang."""
    for i in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            resp.encoding = "utf-8"
            return resp.text
        except requests.RequestException as e:
            print(f"[Lan {i}/{retries}] Loi tai trang: {e}")
            time.sleep(delay)
    raise RuntimeError("Khong tai duoc trang sau nhieu lan thu.")


def parse_number(text):
    """'96.500' -> 96500 ; '+1.800' -> 1800 ; '-200' -> -200 ; '-' -> None."""
    text = text.strip().replace("\xa0", "").replace(" ", "")
    if text in ("", "-", "–", "—"):
        return None
    sign = -1 if text.startswith("-") else 1
    digits = re.sub(r"[^\d]", "", text)
    return sign * int(digits) if digits else None


def parse_table(html):
    """Tim bang co cac cot Ngay / Gia trung binh / Chenh lech."""
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for table in soup.find_all("table"):
        header = table.get_text(" ", strip=True).lower()
        if "ngày" not in header or "chênh lệch" not in header:
            continue
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if len(cells) < 3:
                continue
            m = re.match(r"(\d{2})-(\d{2})-(\d{4})$", cells[0])
            if not m:          # bo qua dong tieu de
                continue
            day, month, year = m.groups()
            rows.append(
                {
                    "date": f"{year}-{month}-{day}",
                    "product": "Cà phê nhân xô",
                    "province": "Đắk Lắk",
                    "price": parse_number(cells[1]),
                    "unit": "đồng/kg",
                    "source": "chogia.vn",
                    "source_url": URL,
                    "price_change": parse_number(cells[2]),
                }
            )
        if rows:
            break
    return rows


def main():
    print("Dang tai du lieu tu:", URL)
    html = fetch_html(URL)
    rows = parse_table(html)
    if not rows:
        raise RuntimeError(
            "Khong tim thay bang gia. Co the trang da doi cau truc HTML, "
            "hay mo trang bang trinh duyet va kiem tra lai the <table>."
        )

    df = pd.DataFrame(rows)
    print(f"Da lay duoc {len(df)} dong, tu {df['date'].min()} den {df['date'].max()}")
    print("Missing:\n", df.isnull().sum())

    # Luu RAW: khong fill, khong xoa duplicate
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print("Da luu file:", OUTPUT)


if __name__ == "__main__":
    main()