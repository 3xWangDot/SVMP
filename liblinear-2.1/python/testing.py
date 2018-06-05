from liblinear import *
from liblinearutil import *
import numpy as np

y, x = svm_read_problem('../heart_scale')
model = train(y, x, '-s 1 -c 1.0 -q')


y=[1.0]*10+[2.0]*10
x=np.random.rand(20,100)
x=x.tolist()
model = train(y, x, '-s 1 -c 1.0 -q')
p_labs, p_acc, p_vals = predict(y, x, model)
[w_k, b_k] = model.get_decfun()
wj = np.asarray(w_k)
w=1
