# CoTMix - Private Target Dataset Generator

Dự án tự động xây dựng private target dataset cho bài toán nhận dạng thể loại nhạc xuyên môi trường. Mục tiêu là tạo ra một tập dữ liệu an toàn về bản quyền, không rò rỉ dữ liệu (data leakage) với tập GTZAN, và đa dạng về môi trường âm thanh.

## Cài đặt

### Yêu cầu hệ thống
- Python 3.9+
- `fpcalc` (công cụ bắt buộc để tính toán fingerprint âm thanh của Chromaprint)

### Hướng dẫn cài đặt
1. **Clone repository và cài đặt thư viện Python:**
   ```bash
   pip install -r requirements.txt