# Đợi từ từ cập nhật nhaaa

# Client 

Đoạn mã này đại diện cho một client trong hệ thống học máy phân tán. Nó thực hiện việc huấn luyện và đánh giá một mô hình CNN bằng cách sử dụng phương pháp học máy phân tán.

### Tập trung công việc chính từ file `client\client_work.py`

## Sử dụng
1. Cài đặt các gói phụ thuộc cần thiết.
2. Chạy đoạn mã bằng cách sử dụng lệnh sau:

## Đầu vào

Đoạn mã client này yêu cầu các đầu vào sau:

1.client\Model_from_Server/model.pt: Tệp mô hình PyTorch được cung cấp bởi máy chủ. Tệp này chứa các tham số mô hình ban đầu mà client sẽ sử dụng trong quá trình huấn luyện.

2.Các tham số cấu hình tùy chỉnh:
- `batch_size`: Kích thước batch cho quá trình huấn luyện mô hình.
- `epochs`: Số epoch để huấn luyện mô hình.
- `percentage_of_dataset`: Phần trăm dữ liệu huấn luyện sẽ được sử dụng.
- `mode`: Chế độ của học máy phân tán (ví dụ: "fedbn").
- `dataset_name`: Tên bộ dữ liệu sẽ được sử dụng.
Các tham số cấu hình khác có thể được truyền qua các đối số dòng lệnh sử dụng `argparse`.

## Đầu ra
Đoạn mã client này tạo ra các đầu ra sau:

1. `client\Model_Client_update/model_client.pt`: Tệp mô hình PyTorch được lưu bởi client sau quá trình huấn luyện. Tệp này chứa các tham số mô hình đã được cập nhật.
2. `eval_list.pkl`: Tệp pickled chứa kết quả đánh giá dưới dạng từ điển được lưu trong file `client\log` trong từng round.

## Luồng làm việc

Đoạn mã client tuân theo các bước sau:

>1.Tải bộ dữ liệu từ thư mục được chỉ định.
>2.Khởi tạo đối tượng client với cấu hình được cung cấp.
>3.Tải các tập dữ liệu vào các trình tải dữ liệu huấn luyện đồng thời tải mô hình kiểm tra mô hình từ server tải về cho vào file `client\Model_from_Server`.
>4.Thiết lập kiến trúc mô hình sử dụng `utils.model.CNNModel`.
>5.Huấn luyện mô hình sử dụng các tập dữ liệu đã tải ở file `client_work.py`.
>6.Đánh giá mô hình đã được huấn luyện ở `client_work.py`.
>7.Lưu mô hình ở dạng `file pytorch (.pt)` vào folder `client\Model_Client_update` và kết quả đánh giá (eval_list) vào tệp `eval_list.pkl` vào `client\log` bằng cách sử dụng `pickle` (lưu ý mỗi client có `eval_list.pkl` mỗi round federated learning).


# Server 

### Tập trung công việc chính từ file `server\server_work.py`

## Đầu vào
Đoạn mã server này yêu cầu các đầu vào sau:

1.Đường dẫn tới mô hình gốc trên server: Đường dẫn đến mô hình được huấn luyện trước trên server `(Server_model_path.pt)`.
2.Đường dẫn tới mô hình `server\log` là nhiều bản eval_list đánh giá của từng client gửi lên : Kịch bản giả định rằng có nhiều client, và mỗi client có một mô hình đã được huấn luyện và một danh sách đánh giá. `server\Model_from_Clients` đường dẫn đến các tệp bỏ model của client vào khi thêm client vào server.

## Đầu ra

Kịch bản này tạo ra các đầu ra sau:
1. Tệp nhật ký: Một tệp nhật ký `(log_dict_round_{fl_round}.pickle)` được tạo ra cho mỗi vòng học phân tán. Nó chứa thông tin như số vòng học phân tán, mất mát huấn luyện theo từng epoch, mất mát đánh giá theo từng epoch, độ chính xác huấn luyện theo từng epoch và độ chính xác đánh giá theo từng epoch.
2. Mô hình toàn cục: Mô hình tổng hợp từ tất cả các client được lưu trữ dưới dạng `global_model.pt` (lưu ở ngay trong file server cũng được) và sau đó file này sẽ được gửi xuống cho các client cập nhật.

## Luồng làm việc

>1.Sử dung `argparse` để config những cái cần thiết cho đầu vào đầu ra như tỷ lệ phần trăm tập dữ liệu được sử dụng để huấn luyện và kiểm tra, tỷ lệ học, kích thước batch, số vòng học phân tán, số epoch trong mỗi local worker, và chế độ `('fedavg' hoặc 'fedbn')`.
>2.Tạo một thể hiện của lớp Server.
>3.Tải mô hình gốc server bằng cách sử dụng đường dẫn mô hình server đã được cung cấp.
>4.Thêm các client vào server bằng cách chỉ định đường dẫn tới các mô hình và danh sách đánh giá của chúng.
>5.Truy cập mô hình từ folder `server\Model_from_Clients` của các client cá nhân bằng cách khởi tạo các client `(ví dụ: client_1 = server[0])`.
>6.Truy xuất danh sách đánh giá và mô hình từ các client từ `folder server\log`.
>7.Tính tỷ lệ số mẫu của client để thực hiện tổng hợp có trọng số.
>8.Tổng hợp các mô hình từ tất cả các client bằng cách sử dụng phương pháp tổng hợp được chỉ định `('fedavg' hoặc 'fedbn')`.
>9.Tính các chỉ số trung bình như mất mát huấn luyện, độ chính xác huấn luyện, mất mát đánh giá và độ chính xác đánh giá.
>10.Lưu trữ từ điển nhật ký là `log_dict` vào folder `server\log_server` chứa các chỉ số cho vòng học phân tán hiện tại.
>11.Lưu trữ mô hình toàn cầu trong ngay folder `server` luôn cũng được (ưng tạo folder chứa cũng được).
>12.In một thông báo xác nhận cho biết mô hình toàn cầu đã được lưu.


## Ghi chú

- Đoạn mã này giả định rằng bộ dữ liệu được tổ chức theo cấu trúc thư mục đã chỉ định.
- Việc sử dụng GPU hỗ trợ CUDA sẽ quyết định xem đoạn mã sử dụng CPU hay GPU để huấn luyện.
- Điều chỉnh đoạn mã theo nhu cầu để phù hợp với cấu hình học máy phân tán cụ thể .
- Có thể thêm các tham số cấu hình hoặc chức năng bổ sung bằng cách sử dụng `argparse`.
- Hãy tùy chỉnh tệp `readme.md` để cung cấp các giải thích chi tiết hơn hoặc bao gồm bất kỳ hướng dẫn cụ thể nào liên quan đến trường hợp sử dụng hoặc yêu cầu .
- Kịch bản này giả định rằng các module và lớp cần thiết `(ClientOnServer, Server)` được định nghĩa trong các tệp riêng biệt `(clients_on_server.py, server.py)` và được nhập vào tương ứng.

