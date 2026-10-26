# English Grammar Correction

Ứng dụng Gradio chạy cục bộ để so sánh suy luận của T5-base và BART-base trên
câu/đoạn văn tiếng Anh. Ứng dụng không huấn luyện mô hình, không ghi đè các
tệp trọng số và không lưu nội dung người dùng vào tệp. Máy chủ chỉ bind vào
`127.0.0.1`; không bật đường dẫn chia sẻ công khai.

## Cấu trúc tài nguyên được kiểm tra

- `model/T5-base.pt`: state dict PyTorch với các khóa T5 như
  `shared.weight`, `encoder.block.*`, `decoder.block.*`.
- `model/Bart-base.pt`: state dict PyTorch với các khóa BART như
  `model.shared.weight`, `model.encoder.layers.*`,
  `model.decoder.layers.*`.
- `T5-base-huggingface/T5-base-huggingface/`: config, tokenizer và generation
  config T5 cục bộ.
- `Bart-base-huggingface/Bart-base-huggingface/`: config, tokenizer và
  generation config BART cục bộ.

Config khai báo `T5ForConditionalGeneration`/`model_type=t5` và
`BartForConditionalGeneration`/`model_type=bart`. Loader khởi tạo các kiến trúc
từ config cục bộ, ánh xạ checkpoint bằng `mmap` và nạp `.pt` bằng
`load_state_dict(strict=True, assign=True)` để giảm đỉnh sử dụng RAM. Các tệp
`model.safetensors` có trong thư mục config không được sử dụng làm trọng số.
Không cần tải tài nguyên từ Hugging Face khi chạy ứng dụng.

## Tiền tố và giới hạn thông tin

Trong workspace không có script/config fine-tune xác nhận tiền xử lý hay tiền
tố GEC. `task_specific_params` trong config T5 chỉ nêu các tác vụ pretrained
như summarization/translation; đó không phải bằng chứng rằng mô hình fine-tune
GEC dùng tiền tố tương ứng. Vì vậy ứng dụng **không tự thêm tiền tố** cho mô
hình nào. Mặc định cả hai nhận nguyên văn đầu vào.

Nếu xác minh được tiền tố từ mã huấn luyện, đặt độc lập cho từng mô hình trước
khi chạy, ví dụ trong PowerShell:

```powershell
$env:T5_INPUT_PREFIX = "tiền tố T5 đã xác nhận"
$env:BART_INPUT_PREFIX = "tiền tố BART đã xác nhận"
```

Để không dùng tiền tố, xóa biến môi trường tương ứng hoặc để trống. Cách viết
hoa/thường, chuẩn hóa văn bản hay tiền xử lý khác cũng chưa thể khẳng định nếu
không có mã huấn luyện.

Đầu vào dài hơn độ dài kiến trúc được tokenizer cắt ở giới hạn tương ứng
(T5: 512 token; BART: 1024 token theo config hiện tại). Ứng dụng hiển thị cảnh
báo khi việc cắt xảy ra. Nhiều câu trong cùng một giới hạn được xử lý cùng
nhau như một đoạn văn.

## Cài đặt và chạy trên Windows

Nên dùng Python 3.10–3.12 để tương thích với các wheel PyTorch cho Windows.
Từ terminal VS Code mở tại thư mục dự án:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Nếu máy có CUDA, hãy cài bản PyTorch tương thích với CUDA theo hướng dẫn cài
đặt chính thức của PyTorch trước khi cài các gói còn lại. Nếu không, PyTorch
chạy trên CPU. Hai mô hình được nạp một lần lúc ứng dụng khởi động, dùng
`eval()`/`inference_mode()` và chạy lần lượt để giảm áp lực bộ nhớ GPU.

Mở địa chỉ cục bộ được in ra trong terminal (thường là
`http://127.0.0.1:7860`). Dùng `Ctrl+C` trong terminal để dừng ứng dụng.

## Kiểm tra nhanh

Nhập ví dụ sai ngữ pháp:

> She go to school every day.

Nhấn **Sửa lỗi ngữ pháp**. Ứng dụng hiển thị câu gốc, đầu ra và thời gian riêng
cho T5-base và BART-base. Kết quả thực tế phụ thuộc vào trọng số fine-tune và
tiền xử lý mà các mô hình đã dùng; hai mô hình có thể trả về câu khác nhau và
ứng dụng không xếp hạng độ chính xác của chúng.

## Xử lý lỗi

- Lỗi thiếu checkpoint/config/tokenizer hoặc sai kiến trúc được hiển thị ở
  thông báo tải mô hình và ô kết quả tương ứng.
- Loader dùng `strict=True`; lỗi khóa hoặc kích thước tensor không tương thích
  sẽ được báo cụ thể, không bị bỏ qua bằng `strict=False`.
- Mỗi mô hình được tải và chạy độc lập. Một mô hình lỗi không xóa kết quả hợp
  lệ của mô hình còn lại.
- Nếu nhập văn bản quá dài, kết quả sẽ kèm cảnh báo rằng đầu vào đã bị cắt.