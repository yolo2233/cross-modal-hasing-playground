function [] = train_lssh(hash_len)

load('temp_data/flickr_data');

opts = [];
opts.mu = 0.5;
opts.rho = 0.2;
opts.lambda = 1;
opts.maxOutIter = 20;

[B,PX,PT,R,A,S,opts]= solveLSSH(train_image',train_text',hash_len,opts);
[train_hash,query_image_hash, query_text_hash] = LSSHcoding(A, B,PX, PT,R,train_image',train_text',query_image,query_text,opts, hash_len);
[train_hash,retrieval_image_hash, retrieval_text_hash] = LSSHcoding(A, B,PX, PT,R,train_image',train_text',retrieval_image,retrieval_text,opts, hash_len);

save('temp_data/hash_code.mat', 'query_image_hash', 'query_text_hash', 'retrieval_image_hash', 'retrieval_text_hash')

end