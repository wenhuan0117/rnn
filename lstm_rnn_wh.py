import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

BATCH_START = 0     # 建立 batch data 时候的 index
TIME_STEPS = 10     # backpropagation through time 的 time_steps
BATCH_SIZE = 100   
INPUT_SIZE = 1      # 输入 size
OUTPUT_SIZE = 1     # 输出 size
CELL_SIZE = 1      # 一个RNN的输出个数，可以自己定义
LR = 0.01          # learning rate


x1=np.linspace(-2,2,500)[:,np.newaxis]*np.pi
y1=np.sin(x1)

def get_batch_data(BATCH_SIZE):
    x=[]
    x_data=[]
    y_data=[]
    for i in range(BATCH_SIZE):
        xx=x1[i:i+TIME_STEPS]
        x_batch=y1[i:i+TIME_STEPS]
        y_batch=y1[i+1:i+TIME_STEPS+1]

        x.append(xx)
        x_data.append(x_batch)
        y_data.append(y_batch)
    return np.array(x),np.array(x_data),np.array(y_data)



def add_layer(inputs,in_size,out_size,activation_function=None):
    weights=tf.Variable(tf.random_normal([in_size,out_size],stddev=1.0))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,weights)+biases

    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)

    return outputs

##定义数据

x,x_data,y_data=get_batch_data(BATCH_SIZE)

xs=tf.placeholder(tf.float32,[None,TIME_STEPS,INPUT_SIZE])


##定义神经层

##l1=add_layer(xs,1,30,activation_function=tf.nn.relu)

##CELL_SIZE不需要与timesteps相等，表示输出维数

lstm_cell = tf.contrib.rnn.BasicLSTMCell(CELL_SIZE, forget_bias=1.0, state_is_tuple=True)

##SIZE必须为BATCH_size
cell_init_state = lstm_cell.zero_state(BATCH_SIZE, dtype=tf.float32)

##xs必须为三维
cell_outputs,cell_final_state = tf.nn.dynamic_rnn(lstm_cell, xs, initial_state=cell_init_state, time_major=False)

output=tf.reshape(cell_outputs,[BATCH_SIZE,TIME_STEPS])
##
prediction1=add_layer(output,TIME_STEPS,TIME_STEPS,activation_function=None)

prediction=tf.reshape(prediction1,[BATCH_SIZE,TIME_STEPS,1])
#
loss=tf.reduce_mean(tf.reduce_sum(tf.square(prediction-y_data),axis=1))

train_op=tf.train.GradientDescentOptimizer(LR).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    plt.ion()
    plt.show()
    for i in range(200):
                
        if i==0:
            _,loss_data,state=sess.run([train_op,loss,cell_final_state],feed_dict={xs:x_data})
            print(i,loss_data,state)
        else:
            _,loss_data,state=sess.run([train_op,loss,cell_final_state],feed_dict={xs:x_data,cell_init_state:state})
            print(i,loss_data,state)
        if i%10==0:
            cell_outputs1,cell_final_state1,pre=sess.run([cell_outputs,cell_final_state,prediction],feed_dict={xs:x_data})
        try:
            ax.lines.remove(lines2[0])
        except Exception:
            pass
            
        lines1=ax.plot(x.flatten(), y_data.flatten(), 'r')
        lines2=ax.plot(x.flatten(), pre.flatten(), 'b--')
        plt.ylim((-1.2, 1.2))
        
##        plt.draw() 
        plt.pause(0.1)

            
