# CS313.L11.KHCL
1. Preprocessing:
    * **Balance**: can bang 2 nhan cua dataset.
    * **Transform**: dung pp minmax de chuan hoa tap du lieu train và dua ve dang One hot Encoding.
    * **TransformImport**: dung pp minmax de chuan hoa tap du lieu upload lên server stream và dua ve dang One hot Encoding.
    * **PCA**: rút trích dac trung tu 22 feature thanh 16 feature.
    * **PCAImport**: rút trich dac trung file dataset duoc upload len server.
2. Train_test_split:
    * Drop 'Unnamed: 0','id' trong train.csv.
    * Bỏ uotlier giá trị extremely outlier lớn hơn 3 lần InterQuartile.
    * Cân bằng 2 nhãn của tập dữ liệu.
    * Xử lí chuẩn hóa minmax và đưa dữ liệu về dạng OneHot.
    * pre.<Hàm> gọi hàm được viết trong file Preprocessing.py
    * chia tập train, test với train_size = 70%.
    * dùng Pickle để lưu lại tập dữ liệu train, test để sử dụng train cho model ML.
3. main:
    * dùng pickle để load 2 model Logistic đã được train sẵn.
    * từ dòng 57 --> :  