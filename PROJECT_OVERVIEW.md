# Tổng quan project

## Mục đích

Ứng dụng Flask hỗ trợ luyện tiếng Anh qua hội thoại thích ứng. Giao diện tập
trung vào lịch sử chat, ô trả lời, kết thúc phiên và các thông tin tiến độ quan
trọng: chủ đề, trình độ, số lượt, điểm ngữ pháp và điểm từ vựng.

## Luồng hội thoại

1. Khi mở trang, trình duyệt tự tạo phiên mới với một câu hỏi ngẫu nhiên trong
   danh sách câu hỏi mở đầu có sẵn.
2. Người học gửi câu trả lời bằng Enter hoặc Send. Giao diện khóa thao tác gửi
   trong khi xử lý.
3. Model 2 kiểm tra câu. Nếu model không thay đổi nội dung, ứng dụng giữ nguyên
   câu gốc và hiển thị lời khích lệ có sẵn; nếu có thay đổi, câu sửa được trình
   bày bằng mẫu “Did you mean: …?”.
4. Model 3 nhận câu đã sửa, chủ đề, trình độ và lịch sử phù hợp để tạo đoạn
   mở rộng ngắn.
5. Model 1 nhận câu gốc/câu sửa, ngữ cảnh mở rộng, lịch sử và câu hỏi trước đó
   để sinh các câu tiếp theo. Conversation manager loại câu hỏi trùng lặp; nếu
   model không thể sinh câu hợp lệ, lỗi được trả rõ ràng thay vì dùng câu mẫu.
6. Người học có thể kết thúc phiên; giao diện hiển thị tổng kết ngắn gọn.

Phản hồi hiển thị trong giao diện không được đưa vào prompt model. Câu trả lời
gốc, câu trả lời đã sửa, đoạn mở rộng và phản hồi được lưu riêng theo từng
lượt. Điểm từ vựng là chỉ số độ đa dạng từ được tính từ văn bản, không phải đầu
ra của một mô hình đánh giá từ vựng riêng.

## Ba mô hình T5 và đường dẫn

`model_loader.py` nạp model/tokenizer từ đĩa cục bộ một lần và tái sử dụng trong
suốt tiến trình:

| Model | Thư mục model và tokenizer |
| --- | --- |
| Model 1 – Question Generation | `T5_Question_Generation/model` |
| Model 2 – Check Grammar | `T5_Check_Grammar` |
| Model 3 – Contextual Text Generation | `T5_Contextual_Text_Generation/fine_tuned_model` |

Mỗi thư mục cần có cấu hình T5, tokenizer và `model.safetensors`. Ứng dụng
chạy trên CUDA nếu PyTorch nhận ra GPU, nếu không thì dùng CPU. API health báo
số model đã nạp, trạng thái từng model và lỗi tải. Nếu thiếu bất kỳ model nào,
chat sẽ báo lỗi dịch vụ; không có kết quả giả để thay thế.

## API

| API | Chức năng |
| --- | --- |
| `GET /` | Giao diện chatbot và tự khởi tạo phiên. |
| `GET /api/health` | Trạng thái ứng dụng và ba model. |
| `POST /api/new-chat` | Tạo phiên và sinh câu hỏi đầu tiên. |
| `POST /api/submit-answer` | Chạy lần lượt kiểm tra ngữ pháp, mở rộng ngữ cảnh và sinh câu tiếp theo. |
| `POST /api/end-chat` | Kết thúc phiên và trả tổng kết. |
| `POST /api/check-grammar` | API riêng để sửa văn bản bằng Model 2. |
| `POST /api/generate-questions` | API riêng để sinh câu hỏi bằng Model 1. |
| `GET /api/chat` | Trạng thái và lịch sử phiên hiện tại. |

## Tệp chính

- `app.py`: Flask routes, tích hợp ba model, phản hồi và health check.
- `model_loader.py`: kiểm tra đường dẫn, nạp model cục bộ và suy luận.
- `conversation_manager.py`: lịch sử, mức độ khó, điều phối lượt và chống lặp.
- `templates/index.html`: giao diện chat tối giản và tương thích màn hình nhỏ.
- `test_conversation_manager.py`: kiểm thử logic hội thoại và API với model stub.
- `test_check_grammar/run.py`, `test_generated_question/run.py`: script chạy thử
  riêng hai model tương ứng.

## Chạy và kiểm thử

Từ thư mục gốc project:

```powershell
python app.py
```

Mở `http://127.0.0.1:5000`. Chạy kiểm thử:

```powershell
python -m unittest -v test_conversation_manager.py
```

## Giới hạn

- Chất lượng đầu ra phụ thuộc dữ liệu huấn luyện và cấu hình của các checkpoint.
- Điểm ngữ pháp là so sánh văn bản gốc với đầu ra sửa của Model 2; điểm từ vựng
  chỉ ước lượng độ đa dạng từ, không thay thế đánh giá trình độ chính thức.
- Phiên lưu trong bộ nhớ của tiến trình Flask, không bền vững qua lần khởi động
  lại và không chia sẻ giữa nhiều worker.
- Ba checkpoint có kích thước lớn; cần đủ RAM/VRAM để nạp và suy luận đồng thời.
