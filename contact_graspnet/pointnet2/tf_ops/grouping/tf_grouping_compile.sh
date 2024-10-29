#/bin/bash
#/usr/local/cuda-8.0/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
/usr/local/cuda-11.6/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
# TF1.2
#g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0


g++ -std=c++14 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I /root/anaconda3/envs/sgbot/lib/python3.8/site-packages/tensorflow/include -I /usr/local/cuda-11.6/include -I /root/anaconda3/envs/sgbot/lib/python3.8/site-packages/tensorflow/include/external/nsync/public  -lcudart -L /usr/local/cuda-11.6/lib64/ -L /root/anaconda3/envs/sgbot/lib/python3.8/site-packages/tensorflow -l:libtensorflow_framework.so.2 -O2 -D_GLIBCXX_USE_CXX11_ABI=0

#g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I /root/anaconda3/envs/sgbot/lib/python3.8/site-packages/tensorflow/include -I /usr/local/cuda-11.6/include -lcudart -L /usr/local/cuda-11.6/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0
# TF1.4
#g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -I /usr/local/lib/python2.7/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-8.0/lib64/ -L/usr/local/lib/python2.7/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0


#/bin/bash
#/usr/local/cuda-10.1/bin/nvcc -std=c++11 -c -o tf_sampling_g.cu.o tf_sampling_g.cu -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
 
# TF1.2
# g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I /home/lib/python2.7/site-packages/tensorflow_core/include -I /usr/local/cuda-10.1/include -lcudart -L /usr/local/cuda-10.1/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0
 
# TF1.15
#g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I /home/lib/python2.7/site-packages/tensorflow_core/include -I /usr/local/cuda-10.1/include -I /home/lib/python2.7/site-packages/tensorflow_core/include/external/nsync/public -lcudart -L /usr/local/cuda-10.1//lib64/ -L/home/lib/python2.7/site-packages/tensorflow_core -l:libtensorflow_framework.so.2 -O2 -D_GLIBCXX_USE_CXX11_ABI=0