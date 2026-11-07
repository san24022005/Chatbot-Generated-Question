# Xây dựng chatbot thông minh sử dụng mô hình ngôn ngữ lớn hỗ trợ đánh giá năng lực tiếng Anh thông qua tương tác hỏi đáp thích ứng

## Giới thiệu

Đề tài hướng tới xây dựng chatbot sử dụng mô hình ngôn ngữ lớn để đánh giá
năng lực tiếng Anh thông qua tương tác hỏi đáp thích ứng. Ứng dụng web hiện
tại cho phép người dùng trả lời câu hỏi tiếng Anh, xem điểm ngữ pháp/từ vựng,
nhận góp ý và câu gợi ý sửa lỗi.

**Tình trạng triển khai:** ứng dụng sử dụng ba mô hình T5 cục bộ theo luồng
sửa ngữ pháp → mở rộng ngữ cảnh → tạo câu hỏi tiếp theo. Cần có đủ ba model
và tokenizer trong các thư mục được liệt kê bên dưới; ứng dụng báo lỗi thay vì
thay kết quả model bằng dữ liệu mẫu.

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
├── model_loader.py                     # Nạp và dùng cả ba mô hình T5
├── conversation_manager.py             # Trạng thái và điều phối hội thoại
├── requirements.txt                    # Các thư viện Python
├── test_check_grammar/
│   └── run.py                          # Chạy thử model sửa ngữ pháp
├── test_generated_question/
│   └── run.py                          # Chạy thử model tạo câu hỏi
├── templates/
│   └── index.html                      # Giao diện chatbot
├── T5_Check_Grammar/
│   └── config.json, tokenizer files, model.safetensors
├── T5_Question_Generation/model/
│   └── config.json, tokenizer files, model.safetensors
└── T5_Contextual_Text_Generation/fine_tuned_model/
    └── config.json, tokenizer files, model.safetensors
```

Mỗi thư mục model phải chứa `config.json`, tokenizer (`tokenizer.json` và/hoặc
`tokenizer_config.json`) và trọng số `model.safetensors`. Các tệp model lớn
không được lưu trong Git. `model_loader.py` nạp cả ba model một lần và giữ
chúng trong bộ nhớ để dùng lại.

## Luồng xử lý

1. Mỗi phiên bắt đầu bằng một câu hỏi ngẫu nhiên trong danh sách câu hỏi có sẵn.
2. Model Check Grammar sửa câu trả lời nếu cần; câu không đổi được giữ nguyên
   và người học nhận lời khích lệ có sẵn.
3. Model Contextual Text Generation mở rộng câu đã sửa thành một đoạn ngắn.
4. Model Question Generation nhận lịch sử, câu đã sửa, đoạn mở rộng và các câu
   đã hỏi để tạo các câu tiếp theo; manager từ chối câu bị lặp.

Điểm ngữ pháp phản ánh việc model có đề xuất sửa câu hay không; điểm từ vựng
là một chỉ số đơn giản về độ đa dạng từ. Đây là tín hiệu luyện tập, không phải
đánh giá chuyên môn hoặc điểm chuẩn hóa theo CEFR.

## Kiểm thử riêng từng mô hình

Các script riêng nạp model từ đúng thư mục cục bộ. Chạy lệnh tại thư mục gốc
dự án, trong môi trường đã cài thư viện:

Kiểm thử mô hình sửa lỗi ngữ pháp

```powershell
python .\test_check_grammar\run.py --text "Text"
```
Ví dụ:

```powershell
python .\test_check_grammar\run.py --text "I usually go home, have dinner with my family, and do my homework. In my free time, I enjoy listening to music, watching movies, and learning new things."
```

Kiểm thử mô hình sinh câu hỏi

```powershell
python .\test_generated_question\run.py --text "Text"
```
Ví dụ:

```powershell
python .\test_generated_question\run.py --text "I once visited Da Lat with my family. We enjoyed the cool weather, beautiful scenery, and delicious food. It was an unforgettable trip."
```

Bỏ `--text "..."` để nhập một đoạn văn bản trực tiếp khi chương trình hỏi.
Kết quả lần lượt là văn bản đã sửa lỗi ngữ pháp và câu hỏi được tạo từ đoạn
văn bản. Các lệnh yêu cầu tệp model và tokenizer tương ứng có sẵn; lỗi nạp
model sẽ được hiển thị thay vì chuyển sang kết quả dự phòng.

Model sửa ngữ pháp nhận tiền tố tác vụ `grammar correction:` trước văn bản đầu
vào để tránh tự ý bỏ nội dung khi trả kết quả.

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

### Cách 1: Chạy trực tiếp trong môi trường ảo (khuyến nghị)

Mở PowerShell, di chuyển đến thư mục gốc của dự án (nơi có tệp `app.py`) rồi
chạy:

```powershell
cd "D:\đường-dẫn\đến\Chatbot-generated-question"
.\.venv\Scripts\python.exe app.py
```

Cách này không cần kích hoạt môi trường ảo trước. Khi máy chủ khởi động thành
công, terminal sẽ hiển thị địa chỉ truy cập mặc định `http://127.0.0.1:5000`
hay `http://localhost:5000`.

### Cách 2: Kích hoạt môi trường ảo trước

Trong PowerShell, kích hoạt `.venv` và chạy bằng lệnh ngắn gọn:

```powershell
cd "D:\đường-dẫn\đến\Chatbot-generated-question"
.\.venv\Scripts\Activate.ps1
python app.py
```

Trước khi chạy, kiểm tra terminal đã trỏ đúng Python trong `.venv` bằng:

```powershell
Get-Command python
```

Đường dẫn kết quả phải bắt đầu bằng `D:\đường-dẫn\đến\Chatbot-generated-question\.venv\Scripts\`.

### Truy cập và dừng ứng dụng

Mở [http://127.0.0.1:5000](http://127.0.0.1:5000) trên trình duyệt. Để kiểm tra
tình trạng máy chủ và các mô hình đã nạp, truy cập
[http://127.0.0.1:5000/api/health](http://127.0.0.1:5000/api/health). Khi cần
dừng ứng dụng, nhấn `Ctrl+C` trong terminal đang chạy `app.py`. Sau đó, nếu đã
kích hoạt môi trường ảo, có thể thoát bằng lệnh `deactivate`.

## Xử lý sự cố

- Nếu chạy `.\.venv\Scripts\python.exe app.py` nhưng nhận `No pyvenv.cfg file`,
  môi trường ảo `.venv` bị thiếu hoặc không hoàn chỉnh. Từ thư mục gốc dự án,
  đổi tên thư mục cũ để giữ lại trước, rồi tạo lại môi trường và cài thư viện:

  ```powershell
  Rename-Item .venv .venv-old
  py -3.11 -m venv .venv
  .\.venv\Scripts\python.exe -m pip install --upgrade pip
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt
  .\.venv\Scripts\python.exe app.py
  ```

  Nếu lệnh `py -3.11` không tìm thấy Python 3.11, kiểm tra các phiên bản đã cài
  bằng `py -0p` và thay `3.11` bằng phiên bản Python 3.10–3.12 có trên máy.
- Nếu Windows báo `Python was not found` khi chạy `python app.py`, không chạy
  lệnh bằng Python của Windows Store. Dùng launcher `py -3.11` để tạo môi trường
  ảo, sau đó chạy `\.venv\Scripts\python.exe app.py`. Nếu đã kích hoạt `.venv`,
  kiểm tra lại bằng `Get-Command python` để xác nhận lệnh trỏ tới
  `.venv\Scripts\python.exe`.
- Nếu PowerShell chặn `Activate.ps1` do chính sách thực thi, không cần kích hoạt
  môi trường: có thể chạy trực tiếp bằng `\.venv\Scripts\python.exe app.py`.
- Nếu giao diện báo không tải được mô hình, kiểm tra `model.safetensors`,
  `config.json` và các tệp tokenizer có trong đúng thư mục như cấu trúc dự án.
- Nếu không nạp được mô hình, API sẽ trả lỗi thay vì thay kết quả bằng luật
  đơn giản. Xem `/api/health` để biết mô hình nào thiếu và nội dung `model_errors`.
- Nếu `POST /api/submit-answer` trả về 503, kiểm tra nội dung `error` trong
  phản hồi và traceback ở terminal. Khi mô hình tạo câu hỏi không hợp lệ hoặc
  trùng lặp, ứng dụng sẽ dùng câu hỏi dự phòng và ghi cảnh báo vào terminal.
- Nếu Windows báo `The paging file is too small` khi chạy một lệnh kiểm thử,
  hãy đóng các ứng dụng khác hoặc tăng kích thước paging file rồi thử lại;
  model cần đủ bộ nhớ khả dụng để nạp trọng số và chạy suy luận.
- Đảm bảo đã cài thư viện trong đúng môi trường ảo đang dùng để chạy `app.py`.
