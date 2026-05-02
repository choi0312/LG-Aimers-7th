# Data Schema

This document summarizes the expected data layout for the LG Aimers 7th F&B menu demand forecasting task.

## Directory Layout

```text
data/
├── train/
│   ├── train.csv
│   ├── price.csv
│   ├── room_type.csv
│   ├── Map.jpg
│   └── meta/
│       ├── TRAIN_group.csv
│       ├── TRAIN_hwadam.csv
│       ├── TRAIN_room.csv
│       ├── TRAIN_ski.csv
│       └── TRAIN_weather.csv
├── test/
│   ├── TEST_00.csv ... TEST_09.csv
│   └── meta/
│       ├── TEST_group_00.csv ... TEST_group_09.csv
│       ├── TEST_hwadam_00.csv ... TEST_hwadam_09.csv
│       ├── TEST_room_00.csv ... TEST_room_09.csv
│       ├── TEST_ski_00.csv ... TEST_ski_09.csv
│       └── TEST_weather_00.csv ... TEST_weather_09.csv
└── sample_submission.csv
```

## Notes

- Raw competition data should not be committed to GitHub.
- The pipeline searches files recursively, so the exact root can be passed through `--data-root`.
- `sample_submission.csv` is required to align final prediction columns and row labels.
