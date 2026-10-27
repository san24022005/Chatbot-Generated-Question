# Xây dựng chatbot thông minh sử dụng mô hình ngôn ngữ lớn hỗ trợ đánh giá năng lực tiếng Anh thông qua tương tác hỏi đáp thích ứng

## Giới thiệu

Đề tài hướng tới xây dựng chatbot sử dụng mô hình ngôn ngữ lớn để đánh giá
năng lực tiếng Anh thông qua tương tác hỏi đáp thích ứng. Ứng dụng web hiện
tại cho phép người dùng trả lời câu hỏi tiếng Anh, xem điểm ngữ pháp/từ vựng,
nhận góp ý và câu gợi ý sửa lỗi.

> **Tình trạng triển khai:** phiên bản hiện tại là bản thử nghiệm. Câu hỏi được
> chọn tuần tự từ một danh sách có sẵn; điểm và một phần nhận xét được tính
> bằng luật đơn giản. Ứng dụng có nạp mô hình T5 kiểm tra ngữ pháp nếu tệp mô
> hình có đầy đủ. Mô hình T5 tạo câu hỏi hiện chưa được gọi trong luồng chạy
> ứng dụng.

## Nguồn dữ liệu

- [Cambridge Learner Corpus (CLC) – FCE dataset](https://researchdatasets.cambridge.org/datasets/clc-fce-dataset):
  dữ liệu người học tiếng Anh ở cấp độ FCE, phục vụ nghiên cứu và phát triển
  các chức năng liên quan đến đánh giá/sửa lỗi.
- [DailyDialog trên Hugging Face](https://huggingface.co/datasets/DeepPavlov/daily_dialog):
  hội thoại tiếng Anh hằng ngày, có thể dùng để tham khảo hoặc phát triển dữ
  liệu hội thoại cho chatbot.

Hãy xem điều khoản truy cập và giấy phép tại trang của từng bộ dữ liệu trước
khi tải hoặc sử dụng. Ứng dụng hiện tại không tự tải hay nạp hai bộ dữ liệu
này khi khởi chạy.

## Cấu trúc dự án

```text
.
├── app.py                              # Ứng dụng web Flask và API đánh giá
├── model_loader.py                     # Nạp mô hình T5 kiểm tra ngữ pháp
├── requirements.txt                    # Các thư viện Python
├── templates/
│   └── index.html                      # Giao diện chatbot
├── T5-check-grammar-model/
│   ├── config.json
│   ├── generation_config.json
│   └── model.safetensors               # Trọng số T5 kiểm tra ngữ pháp
├── T5-check-grammar-tokenizer/
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── T5-question-generated-model/
│   ├── config.json
│   ├── generation_config.json
│   └── model.safetensors               # Trọng số T5 tạo câu hỏi
└── T5-question-generated-tokenizer/
    ├── tokenizer.json
    └── tokenizer_config.json
```

## Tải các mô hình

Tải hoặc cập nhật tệp `model.safetensors` từ các thư mục Google Drive sau:

1. **T5 tạo câu hỏi**: [Google Drive](https://drive.google.com/drive/folders/1MGY-7CC3iCZfHfkbNU97Pr9ozU9PjDSI?usp=drive_link)
2. **T5 kiểm tra ngữ pháp**: [Google Drive](https://drive.google.com/drive/folders/15R-zuBeOkUwoRQKcwpgYMGu5OxQSmpxk?usp=drive_link)

Trong Google Drive, mở thư mục tương ứng và tải `model.safetensors`. Chép
tệp vào đúng thư mục mô hình trong thư mục dự án, giữ nguyên tên tệp:

```text
T5-question-generated-model/model.safetensors
T5-check-grammar-model/model.safetensors
```

Giữ lại các tệp `config.json` và `generation_config.json` đi kèm trong mỗi
thư mục mô hình; không thay chúng bằng tệp trọng số. Tokenizer nằm trong các
thư mục tokenizer riêng như cấu trúc ở trên. Mô hình tạo câu hỏi hiện chưa
được ứng dụng nạp; mô hình kiểm tra ngữ pháp được nạp khi bắt đầu phiên chat.

## Cài đặt thư viện

Yêu cầu Python 3.10–3.12. Ví dụ dưới đây dùng PowerShell trên Windows, chạy
các lệnh tại thư mục dự án:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Các thư viện chính được cài từ `requirements.txt`:

- Flask: máy chủ web và API.
- PyTorch: chạy mô hình.
- Transformers: nạp mô hình T5, tokenizer và sinh đầu ra.

Nếu không dùng PowerShell hoặc không muốn kích hoạt môi trường ảo, có thể gọi
trực tiếp Python trong môi trường ảo:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

PyTorch sẽ tự chọn CUDA nếu môi trường cài đặt hỗ trợ; nếu không, mô hình chạy
bằng CPU. Việc suy luận bằng CPU có thể chậm hơn.

## Chạy chương trình

Từ thư mục dự án, kích hoạt môi trường ảo (nếu chưa kích hoạt) và chạy:

```powershell
python app.py
```

Mở [http://127.0.0.1:5000](http://127.0.0.1:5000) trên trình duyệt. Để kiểm tra
tình trạng máy chủ và mô hình, truy cập
[http://127.0.0.1:5000/api/health](http://127.0.0.1:5000/api/health).
Dừng chương trình bằng `Ctrl+C` trong terminal.

## Xử lý sự cố

- Nếu giao diện báo không tải được mô hình, kiểm tra `model.safetensors`,
  `config.json` và các tệp tokenizer có trong đúng thư mục như cấu trúc dự án.
- Nếu không nạp được mô hình, ứng dụng vẫn có chế độ dự phòng dựa trên luật
  đơn giản; kết quả chế độ này không tương đương với đánh giá bằng mô hình AI.
- Đảm bảo đã cài thư viện trong đúng môi trường ảo đang dùng để chạy `app.py`.
