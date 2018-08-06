# 在keras中使用tf.dataset作为输入

tf.dataset的API支持keras，是Tensorflow-1.9.0的一个新特性。使用tf.dataset作为输入的pipeline可以减少系统内存和显存的占用率，使得在训练大规模数据时，避免内存容量不够的问题出现。

在tensorflow的官方文档中，有简单的[使用说明](https://www.tensorflow.org/guide/keras)，但仍有一些[bug](https://github.com/tensorflow/tensorflow/issues/20827)在tensorflow-1.10.0中才被修复。针对这个版本尚存在的一些问题，以下介绍一些可以规避这些问题的使用办法。

将tf.dataset作为输入传入可以在fit()函数中，也可以在Input()层和compiler()函数中分别传入。
 ``` python
#在fit()函数传入
model.fit(x=iter_x.get_next(),y=iter_y.get_next(),
          epochs=epochs,steps_per_epoch=steps_per_epoch)
# 在Input()层和compiler()函数传入
inputs = Input(tensor=iter_x.get_next())
model.compile(loss=loss,optimizer=optimizer,
              target_tensors=[iter_y.get_next()])
```
传入时需要将dataset类转换为tensor类。这一步涉及两个步骤，首先需要生成dataset的iterator,对于不同的dataset有不同生成方法，常用的有one_shot和initializable。之后通过iterator的get_next()函数迭代获取tensor类数据。
``` python
iterator = dataset.make_initializable_iterator()
iterator = dataset.make_one_shot_iterator()
iterator = iterator.get_next()
```
keras对于传入的tensor有tf.dtype的要求，x需要是tf.float32类型。如果类型不符，可以通过tf.cast()进行类型转换
``` python
inputs = Input(tensor=tf.cast(iter_x.get_next(),tf.float32))
```

面对一系列初始化的需要时，可以先获取kera的session，并在keras的session中对table和iterator进行初始化
``` python
from keras import backend as K
K.get_session().run(tf.tables_initializer())
K.get_session().run(iter_data.initializer)
```
