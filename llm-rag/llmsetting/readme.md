Khi thiết kế và thử nghiệm prompt, bạn thường tương tác với LLM (Mô hình Ngôn ngữ Lớn) thông qua một API. Bạn có thể cấu hình một số tham số để nhận được các kết quả khác nhau cho các prompt của mình. Điều chỉnh các cài đặt này là quan trọng để cải thiện độ tin cậy và chất lượng của các phản hồi, và cần một số thử nghiệm để tìm ra cài đặt phù hợp cho các trường hợp sử dụng của bạn. Dưới đây là các cài đặt phổ biến bạn sẽ gặp khi sử dụng các nhà cung cấp LLM khác nhau:

Nhiệt độ (Temperature) - Nói ngắn gọn, càng thấp nhiệt độ, kết quả càng mang tính xác định theo nghĩa là token có khả năng cao nhất luôn được chọn. Tăng nhiệt độ có thể dẫn đến tính ngẫu nhiên cao hơn, khuyến khích các đầu ra đa dạng hoặc sáng tạo hơn. Về cơ bản, bạn đang tăng trọng số của các token khác có thể. Trong ứng dụng thực tế, bạn có thể muốn sử dụng giá trị nhiệt độ thấp hơn cho các nhiệm vụ như trả lời câu hỏi dựa trên sự kiện để khuyến khích các phản hồi chính xác và ngắn gọn. Đối với việc tạo thơ hoặc các nhiệm vụ sáng tạo khác, việc tăng giá trị nhiệt độ có thể mang lại lợi ích.

Top P - Một kỹ thuật lấy mẫu đi kèm với nhiệt độ, được gọi là lấy mẫu hạt nhân, cho phép bạn kiểm soát mức độ xác định của mô hình. Nếu bạn muốn các câu trả lời chính xác và dựa trên sự kiện, hãy giữ giá trị này thấp. Nếu bạn muốn các phản hồi đa dạng hơn, hãy tăng lên một giá trị cao hơn. Khi sử dụng Top P, nghĩa là chỉ các token chiếm khối lượng xác suất top_p mới được xem xét cho các phản hồi, do đó một giá trị top_p thấp sẽ chọn các phản hồi tự tin nhất. Điều này có nghĩa là một giá trị top_p cao sẽ cho phép mô hình xem xét nhiều từ khả dĩ hơn, bao gồm cả những từ ít có khả năng, dẫn đến các đầu ra đa dạng hơn.

Khuyến nghị chung là thay đổi nhiệt độ hoặc Top P, nhưng không nên thay đổi cả hai.

Độ dài tối đa (Max Length) - Bạn có thể quản lý số lượng token mà mô hình tạo ra bằng cách điều chỉnh độ dài tối đa. Việc chỉ định độ dài tối đa giúp bạn ngăn chặn các phản hồi dài hoặc không liên quan và kiểm soát chi phí.

Chuỗi dừng (Stop Sequences) - Chuỗi dừng là một chuỗi ký tự ngăn mô hình tạo ra các token. Việc chỉ định các chuỗi dừng là một cách khác để kiểm soát độ dài và cấu trúc của phản hồi mô hình. Ví dụ, bạn có thể yêu cầu mô hình tạo ra các danh sách không quá 10 mục bằng cách thêm "11" làm chuỗi dừng.

Mức phạt tần suất (Frequency Penalty) - Mức phạt tần suất áp dụng một mức phạt cho token tiếp theo tỷ lệ với số lần token đó đã xuất hiện trong phản hồi và prompt. Càng cao mức phạt tần suất, từ càng ít có khả năng xuất hiện lại. Cài đặt này giảm thiểu việc lặp lại từ trong phản hồi của mô hình bằng cách áp dụng mức phạt cao hơn cho các token xuất hiện nhiều hơn.

Mức phạt hiện diện (Presence Penalty) - Mức phạt hiện diện cũng áp dụng mức phạt cho các token lặp lại, nhưng không giống như mức phạt tần suất, mức phạt là như nhau cho tất cả các token lặp lại. Một token xuất hiện hai lần và một token xuất hiện 10 lần sẽ bị phạt như nhau. Cài đặt này ngăn mô hình lặp lại các cụm từ quá thường xuyên trong phản hồi của nó. Nếu bạn muốn mô hình tạo ra văn bản đa dạng hoặc sáng tạo, bạn có thể muốn sử dụng mức phạt hiện diện cao hơn. Hoặc, nếu bạn cần mô hình tập trung, hãy thử sử dụng mức phạt hiện diện thấp hơn.

Tương tự như nhiệt độ và top_p, khuyến nghị chung là điều chỉnh mức phạt tần suất hoặc hiện diện, nhưng không nên điều chỉnh cả hai.
