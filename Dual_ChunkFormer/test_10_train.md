# Test 10 câu train (ChunkFormer pretrained, chưa fine-tune, chưa Qwen)

Nguồn: train.jsonl, train_0001–train_0010. Chạy CPU, `--no-llm`.

**WER trung bình (base): 0.2161**

Lưu ý: reference viết số (50, 123, 5 giờ), model ra chữ nên WER bị phạt oan ở câu 6, 7, 9.

## 1. train_0001.wav — WER 0.16
- REF: Nếu mai mưa thì dời chạy bộ muộn 1 tiếng, nhắn Sếp Dũng giờ mới; không mưa thì thôi.
- HYP: nếu mai mưa thì giời chạy bộ muộn một chuyến nhắn sếp dũng giờ mới không mưa thì thôi
- Sentence (slot): nếu mai mưa thì giời chạy bộ muộn một [SLOT_1] nhắn sếp dũng giờ mới không mưa thì thôi

## 2. train_0002.wav — WER 0.00
- REF: Nếu mai mưa thì hủy lịch đá bóng... À mà thôi, cứ để lịch đó, chỉ cần xem giúp mai có mưa không.
- HYP: nếu mai mưa thì hủy lịch đá bóng à mà thôi cứ để lịch đó chỉ cần xem giúp mai có mưa không
- Sentence (slot): nếu mai mưa thì hủy lịch đá bóng à mà thôi cứ để lịch đó chỉ cần xem giúp mai có mưa không

## 3. train_0003.wav — WER 0.26
- REF: Ơi, nạp giúp mình 50 nghìn điện thoại cho bác Ba với chuyển ví cho cô Mai số tiền điện tháng này nhé.
- HYP: ơi nạp giúp mình năm mươi ngàn tiền điện thoại cho bác ba vé chuyển ví cho công mai số tiền điện tháng này nhé
- Sentence (slot): ơi [SLOT_1] giúp mình năm mươi ngàn tiền điện thoại cho bác ba [SLOT_2] chuyển [SLOT_3] cho [SLOT_4] mai số tiền điện [SLOT_5] này nhé

## 4. train_0004.wav — WER 0.00
- REF: Trợ lý ơi, giảm độ sáng đèn phòng ngủ nhỏ xuống còn một nửa nhé.
- HYP: trợ lý ơi giảm độ sáng đèn phòng ngủ nhỏ xuống còn một nửa nhé
- Sentence (slot): trợ lý ơi giảm độ sáng đèn phòng ngủ nhỏ xuống còn một nửa nhé

## 5. train_0005.wav — WER 0.17
- REF: Trợ lý ơi, đặt giúp tôi xe bốn chỗ từ công ty đi họp phụ huynh chiều nay nhé, đặt ngay bây giờ luôn.
- HYP: trợ lý ơi đặt giúp tôi xe vốn chỗ từ công ty đi học của uyên chiều nay nhé đặt ngay bây giờ luôn
- Sentence (slot): trợ lý ơi đặt giúp tôi xe vốn [SLOT_1] từ công ty đi học của [SLOT_2] chiều nay nhé đặt ngay bây giờ luôn

## 6. train_0006.wav — WER 0.43
- REF: Nếu giá mua vào vàng SJC hôm nay dưới 123 triệu thì 9 giờ sáng mai nhắc tôi đi mua nhé.
- HYP: điếu giá thu mua vào vàng sgc hôm nay dưới một trăm hai mươi ba triệu thì chín giờ sáng mai nhắc tôi đi mua nhé
- Sentence (slot): [SLOT_1] giá thu mua vào vàng sgc hôm nay dưới một trăm hai mươi ba triệu thì chín giờ sáng mai nhắc tôi đi mua nhé

## 7. train_0007.wav — WER 0.23
- REF: Đặt báo thức 5 giờ nhé, à không, 5 rưỡi sáng mai đó
- HYP: tập báo thức năm giờ nhé à không năm rưỡi sáng mai đó
- Sentence (slot): tập báo thức năm giờ nhé à không năm rưỡi sáng mai đó

## 8. train_0008.wav — WER 0.13
- REF: Cái còn lại nếu mai mưa thì dời muộn nửa tiếng, không mưa giữ nguyên.
- HYP: cái còn lại nếu mai mưa thì dời muộn nửa tiếng không mưa rượu huyên
- Sentence (slot): cái còn lại nếu mai mưa thì [SLOT_1] muộn nửa tiếng không [SLOT_2] rượu huyên

## 9. train_0009.wav — WER 0.75
- REF: Nạp dùm em Linh 50 ngàn cho số không chín năm không năm ba sáu không ba sáu bảy nha.
- HYP: đặt dùm em linh năm mươi nghìn không trăm cho số không trăm chín mươi lăm không nghìn năm trăm ba mươi sáu không nghìn ba trăm sáu mươi bảy nha
- Sentence (slot): đặt dùm em linh năm mươi nghìn không trăm cho số không [SLOT_1] chín mươi lăm không [SLOT_2] năm trăm ba mươi sáu không nghìn ba trăm sáu mươi bảy nha

## 10. train_0011.wav — WER 0.03
- REF: Nếu ví đủ thì trả điện và nhắn Bố là đã đóng, còn không thì trả ngân hàng và nhắc tôi tám giờ tối nay nạp tiền vào ví nhé.
- HYP: nếu ví đủ thì trả điện và nhắn bố là đã đóng còn không thì trả ngân hàng và nhắc tôi tám giờ tối nay nạp tiền vào ví nha
- Sentence (slot): nếu ví đủ thì trả điện và nhắn bố là đã đóng còn không thì trả ngân hàng và nhắc tôi tám giờ tối nay nạp tiền vào ví nha
