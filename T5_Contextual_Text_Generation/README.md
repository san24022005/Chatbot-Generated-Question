# Mô hình T5 Contextual Text Generation (Fine-tuned trên WoW Dataset)

Mô hình này được phát triển để phục vụ chatbot hỗ trợ học Tiếng Anh. Chức năng chính là mở rộng câu trả lời ngắn của học viên (Learner Answer) dựa theo ngữ cảnh cuộc hội thoại và thông tin kiến thức bổ trợ tương ứng.

## Hướng dẫn cài đặt và chạy trên Windows (VS Code)

1. Giải nén tệp `T5_Contextual_Text_Generation.zip` vào thư mục dự án của bạn.
2. Mở terminal tại thư mục vừa giải nén.
3. Cài đặt các thư viện cần thiết bằng pip:
   ```bash
   pip install -r requirements.txt
   ```
4. Chạy tệp script để kiểm thử trực tiếp:
   ```bash
   python inference.py
   ```

## Cấu trúc thư mục đóng gói:
- `fine_tuned_model/`: Chứa tham số trọng số và cấu hình cấu trúc mạng nơ-ron T5-base.
- `inference.py`: Chứa mã wrapper thực thi mô hình với Beam Search.
- `requirements.txt`: Chứa danh sách phiên bản thư viện đi kèm.
