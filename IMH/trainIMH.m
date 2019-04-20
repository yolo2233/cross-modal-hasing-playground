function [] = trainIMH(training_num, hash_len)

load('temp_data/flickr_data')

para.k=10;
para.sigma=1;
L1=Laplacian_GK(train_img',para);
L2=Laplacian_GK(train_txt',para);

[X1_train, query_img_binary, X2_train, query_txt_binary] = crossMediaTraining(train_img', train_txt', training_num, -3, -3, L1, L2, query_img, query_txt, hash_len);
[X1_train, retrieval_img_binary, X2_train, retrieval_txt_binary] = crossMediaTraining(train_img', train_txt', training_num, -3, -3, L1, L2, retrieval_img, retrieval_txt, hash_len);

size(X1_train)
size(X2_train)
query_img_binary = query_img_binary > 0;
query_txt_binary = query_txt_binary > 0;
retrieval_img_binary = retrieval_img_binary > 0;
retrieval_txt_binary = retrieval_txt_binary > 0;

query_img_binary = compactbit(query_img_binary);
query_txt_binary = compactbit(query_txt_binary);
retrieval_img_binary = compactbit(retrieval_img_binary);
retrieval_txt_binary = compactbit(retrieval_txt_binary);

ham_dis_i2t = hammingDist(query_img_binary,retrieval_txt_binary);
ham_dis_t2i = hammingDist(query_txt_binary,retrieval_img_binary);
size(ham_dis_i2t)
size(ham_dis_t2i)

save('temp_data/hash_code.mat', 'query_img_binary', 'query_txt_binary', 'retrieval_img_binary', 'retrieval_txt_binary')
save('temp_data/ham_dis.mat','ham_dis_i2t', 'ham_dis_t2i')