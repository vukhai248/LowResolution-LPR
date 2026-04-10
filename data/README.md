# Data Directory

Dữ liệu được quản lý bằng **DVC** (Data Version Control) và lưu trữ trên Google Drive.

## Cấu trúc thư mục

```
data/
├── ICPR 2026/                            # Dataset chính (DVC tracked)
│   ├── train/
│   │   ├── Scenario-A/
│   │   │   ├── Brazilian/
│   │   │   └── Mercosur/
│   │   └── Scenario-B/
│   │       ├── Brazilian/
│   │       └── Mercosur/
│   └── test/
│       ├── track_10005/
│       ├── track_10015/
│       ├── ...                           # Nhiều thư mục track_* (không liệt kê hết)
│       └── track_xxxxx/
├── test.zip                              # File nén test (nếu có)
└── README.md                             # File này
```

## 🔽 Cách tải dữ liệu

Có thể tải thông qua dvc hoặc link drive sau: 

### Bước 1: Cài DVC
```bash
pip install dvc dvc-gdrive
```

### Bước 2: Thêm DVC remote (chỉ cần làm 1 lần nếu chưa có)

```bash
# FOLDER_ID lấy từ URL Google Drive:
# https://drive.google.com/drive/folders/FOLDER_ID (FOLDER_ID ở cuối trước dấu //)
dvc remote add -d gdrive gdrive://FOLDER_ID
```

### Bước 3: Cấu hình OAuth credentials

Cần file `client_secret_*.json` (liên hệ team để nhận). Mở file → copy `client_id` và `client_secret`, sau đó chạy:

```bash
# Cách 1: Dùng OAuth credentials (phổ biến)
dvc remote modify --local gdrive gdrive_client_id "CLIENT_ID_TRONG_FILE_JSON"
dvc remote modify --local gdrive gdrive_client_secret "CLIENT_SECRET_TRONG_FILE_JSON"

# Cách 2: Dùng Service Account
dvc remote modify --local gdrive gdrive_use_service_account true
dvc remote modify --local gdrive gdrive_service_account_json_file_path "path/to/service-account.json"
```

### Bước 3: Pull dữ liệu
```bash
dvc pull                  # Tải toàn bộ data
dvc pull data/raw.dvc     # Chỉ tải raw data
```

Trình duyệt sẽ mở để đăng nhập Google (lần đầu). Cần có quyền truy cập folder Drive của project.

## ⚠️ Lưu ý
- **KHÔNG** commit file `client_secret*.json` hoặc `service_account*.json` lên Git
- File `.dvc/config.local` chứa credentials cá nhân, đã nằm trong `.gitignore`
- Khi thay đổi data → chạy `dvc add data/<folder>` → `dvc push` → `git add .` → `git commit`
