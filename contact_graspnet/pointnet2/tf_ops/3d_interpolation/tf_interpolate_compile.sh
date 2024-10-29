# TF1.2
#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /root/anaconda3/envs/sgbot/lib/python3.8/site-packages/tensorflow/include -I /usr/local/cuda-11.6/include -lcudart -L /usr/local/cuda-11.6/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.4
#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -I /usr/local/lib/python2.7/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-8.0/lib64/ -L/usr/local/lib/python2.7/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0


g++ -std=c++14 tf_interpolate.cpp  -o tf_interpolate_so.so -shared -fPIC -I /root/anaconda3/envs/sgbot/lib/python3.8/site-packages/tensorflow/include -I /usr/local/cuda-11.6/include -I /root/anaconda3/envs/sgbot/lib/python3.8/site-packages/tensorflow/include/external/nsync/public  -lcudart -L /usr/local/cuda-11.6/lib64/ -L /root/anaconda3/envs/sgbot/lib/python3.8/site-packages/tensorflow -l:libtensorflow_framework.so.2 -O2 -D_GLIBCXX_USE_CXX11_ABI=0