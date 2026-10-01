import argparse
import csv
import re
import sys
from datetime import date, timedelta
from html.parser import HTMLParser
from urllib.request import Request, urlopen


SOURCE_URL = "https://chogia.vn/gia-ca-phe-dak-lak-hom-nay/"
DEFAULT_OUTPUT = "data/raw/coffee/coffee_daklak_raw12345.csv"
SOURCE_NAME = "Chợ Giá"
PRODUCT_NAME = "Cà phê"
PROVINCE_NAME = "Đắk Lắk"
PRICE_UNIT = "VND/kg"


class PriceTableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_target_table = False
        self.table_depth = 0
        self.in_row = False
        self.in_cell = False
        self.cell_text = []
        self.row = []
        self.rows = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "table" and attributes.get("id") == "cf_df":
            self.in_target_table = True
            self.table_depth = 1
        elif self.in_target_table and tag == "table":
            self.table_depth += 1
        elif self.in_target_table and tag == "tr":
            self.in_row = True
            self.row = []
        elif self.in_row and tag == "td":
            self.in_cell = True
            self.cell_text = []

    def handle_endtag(self, tag):
        if self.in_target_table and tag == "td" and self.in_cell:
            self.row.append("".join(self.cell_text).strip())
            self.in_cell = False
        elif self.in_target_table and tag == "tr" and self.in_row:
            if len(self.row) >= 3 and re.fullmatch(r"\d{2}-\d{2}-\d{4}", self.row[0]):
                self.rows.append(self.row[:3])
            self.in_row = False
        elif self.in_target_table and tag == "table":
            self.table_depth -= 1
            if self.table_depth == 0:
                self.in_target_table = False

    def handle_data(self, data):
        if self.in_cell:
            self.cell_text.append(data)


def parse_integer(value):
    value = value.strip().replace(".", "").replace(",", "")
    if value in {"", "-"}:
        return ""
    sign = -1 if value.startswith("-") else 1
    value = value.lstrip("+-")
    return sign * int(value)


def fetch_rows():
    request = Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        html = response.read().decode(response.headers.get_content_charset() or "utf-8", errors="replace")
    parser = PriceTableParser()
    parser.feed(html)
    rows_by_date = {}
    for day, price, change in parser.rows:
        rows_by_date[date.fromisoformat("-".join(reversed(day.split("-"))))] = {
            "date": date.fromisoformat("-".join(reversed(day.split("-")))),
            "average_price_vnd_per_kg": parse_integer(price),
            "change_vnd_per_kg": parse_integer(change),
        }
    return list(rows_by_date.values())


def main():
    argument_parser = argparse.ArgumentParser(description="Cào giá cà phê Đắk Lắk từ Chợ Giá.")
    argument_parser.add_argument("--output", default=DEFAULT_OUTPUT)
    argument_parser.add_argument("--days", type=int, default=895, help="Số ngày cần kiểm tra, mặc định khoảng 3 năm.")
    args = argument_parser.parse_args()

    rows = fetch_rows()
    end_date = max(row["date"] for row in rows)
    start_date = end_date - timedelta(days=args.days - 1)
    rows = sorted(
        (row for row in rows if start_date <= row["date"] <= end_date),
        key=lambda row: row["date"],
    )

    output_path = args.output
    output_directory = output_path.rsplit("/", 1)[0] if "/" in output_path else output_path.rsplit("\\", 1)[0] if "\\" in output_path else ""
    if output_directory:
        import os

        os.makedirs(output_directory, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=("date", "province", "product", "price", "unit", "source", "price_change", "source_url"),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "date": row["date"].isoformat(),
                    "province": PROVINCE_NAME,
                    "product": PRODUCT_NAME,
                    "price": row["average_price_vnd_per_kg"],
                    "unit": PRICE_UNIT,
                    "source": SOURCE_NAME,
                    "price_change": row["change_vnd_per_kg"],
                    "source_url": SOURCE_URL,
                }
            )

    available_dates = {row["date"] for row in rows}
    expected_dates = {start_date + timedelta(days=offset) for offset in range(args.days)}
    missing_dates = sorted(expected_dates - available_dates)
    print(f"Đã ghi {len(rows)} dòng vào {output_path}")
    print(f"Phạm vi lấy được: {min(available_dates)} đến {max(available_dates)}")
    

if __name__ == "__main__":
    main()