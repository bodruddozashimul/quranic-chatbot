"""
build_dataset.py
----------------
Downloads and merges the Arabic text of the Quran with its English and Bengali
translations into a single flat JSON file used by the ingestion step.

Source: fawazahmed0/quran-api (raw.githubusercontent.com mirror).
Each edition file has the shape:
    {"quran": [{"chapter": 1, "verse": 1, "text": "..."}, ...]}
with exactly 6236 verses in canonical order.

The Bengali edition is Muhiuddin Khan, which is the translation named in the
thesis. The English edition is configurable; Talal Itani is the default because
its phrasing is close to the wording used in the report's screenshots.

Output: /app/data/quran_full.json
"""

import json
import os
import sys
import urllib.request
import urllib.error

RAW = "https://raw.githubusercontent.com/fawazahmed0/quran-api/1/editions"

EDITIONS = {
    "arabic": os.getenv("EDITION_AR", "ara-quranuthmanienc"),
    "english": os.getenv("EDITION_EN", "eng-talalitani"),
    "bengali": os.getenv("EDITION_BN", "ben-muhiuddinkhan"),
}

# Fallbacks tried in order if the primary edition fails to download.
FALLBACKS = {
    "english": ["eng-abdelhaleem", "eng-mohammedmarmadu", "eng-abdullahyusufal"],
    "bengali": ["ben-abubakrzakaria", "ben-zohurulhoque"],
    "arabic": ["ara-quransimple", "ara-quranuthmani"],
}

OUTPUT_PATH = os.getenv("DATASET_PATH", "/app/data/quran_full.json")
EXPECTED_VERSES = 6236

SURAH_NAMES_EN = [
    "Al-Fatihah", "Al-Baqarah", "Ali 'Imran", "An-Nisa", "Al-Ma'idah",
    "Al-An'am", "Al-A'raf", "Al-Anfal", "At-Tawbah", "Yunus", "Hud",
    "Yusuf", "Ar-Ra'd", "Ibrahim", "Al-Hijr", "An-Nahl", "Al-Isra",
    "Al-Kahf", "Maryam", "Ta-Ha", "Al-Anbya", "Al-Hajj", "Al-Mu'minun",
    "An-Nur", "Al-Furqan", "Ash-Shu'ara", "An-Naml", "Al-Qasas", "Al-Ankabut",
    "Ar-Rum", "Luqman", "As-Sajdah", "Al-Ahzab", "Saba", "Fatir", "Ya-Sin",
    "As-Saffat", "Sad", "Az-Zumar", "Ghafir", "Fussilat", "Ash-Shuraa",
    "Az-Zukhruf", "Ad-Dukhan", "Al-Jathiyah", "Al-Ahqaf", "Muhammad", "Al-Fath",
    "Al-Hujurat", "Qaf", "Adh-Dhariyat", "At-Tur", "An-Najm", "Al-Qamar",
    "Ar-Rahman", "Al-Waqi'ah", "Al-Hadid", "Al-Mujadila", "Al-Hashr",
    "Al-Mumtahanah", "As-Saf", "Al-Jumu'ah", "Al-Munafiqun", "At-Taghabun",
    "At-Talaq", "At-Tahrim", "Al-Mulk", "Al-Qalam", "Al-Haqqah", "Al-Ma'arij",
    "Nuh", "Al-Jinn", "Al-Muzzammil", "Al-Muddaththir", "Al-Qiyamah",
    "Al-Insan", "Al-Mursalat", "An-Naba", "An-Nazi'at", "Abasa", "At-Takwir",
    "Al-Infitar", "Al-Mutaffifin", "Al-Inshiqaq", "Al-Buruj", "At-Tariq",
    "Al-A'la", "Al-Ghashiyah", "Al-Fajr", "Al-Balad", "Ash-Shams", "Al-Layl",
    "Ad-Duhaa", "Ash-Sharh", "At-Tin", "Al-Alaq", "Al-Qadr", "Al-Bayyinah",
    "Az-Zalzalah", "Al-Adiyat", "Al-Qari'ah", "At-Takathur", "Al-Asr",
    "Al-Humazah", "Al-Fil", "Quraysh", "Al-Ma'un", "Al-Kawthar", "Al-Kafirun",
    "An-Nasr", "Al-Masad", "Al-Ikhlas", "Al-Falaq", "An-Nas",
]

SURAH_NAMES_AR = [
    "الفاتحة", "البقرة", "آل عمران", "النساء", "المائدة", "الأنعام", "الأعراف",
    "الأنفال", "التوبة", "يونس", "هود", "يوسف", "الرعد", "إبراهيم", "الحجر",
    "النحل", "الإسراء", "الكهف", "مريم", "طه", "الأنبياء", "الحج", "المؤمنون",
    "النور", "الفرقان", "الشعراء", "النمل", "القصص", "العنكبوت", "الروم",
    "لقمان", "السجدة", "الأحزاب", "سبأ", "فاطر", "يس", "الصافات", "ص", "الزمر",
    "غافر", "فصلت", "الشورى", "الزخرف", "الدخان", "الجاثية", "الأحقاف", "محمد",
    "الفتح", "الحجرات", "ق", "الذاريات", "الطور", "النجم", "القمر", "الرحمن",
    "الواقعة", "الحديد", "المجادلة", "الحشر", "الممتحنة", "الصف", "الجمعة",
    "المنافقون", "التغابن", "الطلاق", "التحريم", "الملك", "القلم", "الحاقة",
    "المعارج", "نوح", "الجن", "المزمل", "المدثر", "القيامة", "الإنسان",
    "المرسلات", "النبأ", "النازعات", "عبس", "التكوير", "الانفطار", "المطففين",
    "الانشقاق", "البروج", "الطارق", "الأعلى", "الغاشية", "الفجر", "البلد",
    "الشمس", "الليل", "الضحى", "الشرح", "التين", "العلق", "القدر", "البينة",
    "الزلزلة", "العاديات", "القارعة", "التكاثر", "العصر", "الهمزة", "الفيل",
    "قريش", "الماعون", "الكوثر", "الكافرون", "النصر", "المسد", "الإخلاص",
    "الفلق", "الناس",
]


def _download(edition: str) -> dict:
    url = f"{RAW}/{edition}.json"
    print(f"    GET {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "quran-rag-demo/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_edition(kind: str) -> dict:
    """Fetch one edition, trying fallbacks if the primary fails."""
    candidates = [EDITIONS[kind]] + FALLBACKS.get(kind, [])
    last_error = None
    for edition in candidates:
        try:
            payload = _download(edition)
            verses = payload["quran"]
            if len(verses) != EXPECTED_VERSES:
                raise ValueError(f"expected {EXPECTED_VERSES} verses, got {len(verses)}")
            print(f"    OK  {kind}: {edition} ({len(verses)} verses)")
            # key by (chapter, verse) so a misordered file cannot silently misalign
            return {(v["chapter"], v["verse"]): v["text"].strip() for v in verses}
        except (urllib.error.URLError, KeyError, ValueError, json.JSONDecodeError) as e:
            print(f"    FAIL {edition}: {e}")
            last_error = e
    raise RuntimeError(f"Could not download any {kind} edition. Last error: {last_error}")


def build() -> list:
    print("Downloading Quran editions...")
    arabic = fetch_edition("arabic")
    english = fetch_edition("english")
    bengali = fetch_edition("bengali")

    rows = []
    missing_bn = 0
    for (chapter, verse), ar_text in sorted(arabic.items()):
        bn = bengali.get((chapter, verse), "")
        if not bn:
            missing_bn += 1
            bn = "[বাংলা অনুবাদ পাওয়া যায়নি]"
        rows.append({
            "surah_no": chapter,
            "ayah_no": verse,
            "surah_name_en": SURAH_NAMES_EN[chapter - 1],
            "surah_name_ar": SURAH_NAMES_AR[chapter - 1],
            "arabic_text": ar_text,
            "english_text": english.get((chapter, verse), ""),
            "bengali_text": bn,
        })

    if missing_bn:
        print(f"    WARNING: {missing_bn} verses had no Bengali translation.")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)

    print(f"Dataset written: {len(rows)} ayahs -> {OUTPUT_PATH}")
    return rows


def verify(rows: list) -> bool:
    """Spot-check a few verses used in the evaluation set."""
    index = {(r["surah_no"], r["ayah_no"]): r for r in rows}
    checks = [(2, 153), (2, 261), (39, 53), (17, 23), (14, 7), (2, 183)]
    ok = True
    print("\nSpot-check of evaluation gold verses:")
    for ref in checks:
        row = index.get(ref)
        if not row or not row["arabic_text"] or not row["english_text"]:
            print(f"  MISSING {ref[0]}:{ref[1]}")
            ok = False
            continue
        print(f"  {ref[0]}:{ref[1]}  {row['surah_name_en']:<14} EN: {row['english_text'][:60]}...")
    return ok


if __name__ == "__main__":
    data = build()
    if not verify(data):
        sys.exit(1)
