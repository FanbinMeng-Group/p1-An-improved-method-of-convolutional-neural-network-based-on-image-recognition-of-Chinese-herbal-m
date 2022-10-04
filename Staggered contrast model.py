import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import glob

# 给标签设置编码
imgs_path = glob.glob("E:\\Dataset Chinese Medicine\\*\\*.jpg")

try:
    for one_path in imgs_path:
        img_path = str(one_path)
        image = tf.io.read_file(img_path, 'rb')
        im_jpg = tf.image.is_jpeg(image)
        if not im_jpg:
            os.remove(img_path)
        image = tf.io.decode_image(image)
        image = tf.image.resize(image, [256, 256])
        if image.shape == (256, 256, 4):
            os.remove(img_path)
        if image.shape == (1, 256, 256, 3):
            os.remove(img_path)
        # print(image.shape)

except Exception:
    pass
finally:
    imgs_path = glob.glob('E:\\Dataset Chinese Medicine\\*\\*.jpg')

print("图片路径:", imgs_path[:3])  # 图片的标签值并不是最后一个  而是所属类别的文件名
print("图片长度:", len(imgs_path))
img_p = imgs_path[1]
all_label_names = [img_p.split('\\')[2] for img_p in imgs_path]  # 不能写imgs_path.split   列表没有拆分（split） 只有字符串有
print(type(img_p))  # 列表索引一个的返回值就是元素本身的类型  列表多个索引得到的返回值是列表
print(all_label_names[:3])
print(all_label_names[-3:])
# 映射标签
lable_names = np.unique(all_label_names)  # 取出唯一值
print("所有标签名:", lable_names)
lable_to_index = dict((name, i) for i, name in enumerate(lable_names))  # 将标签进行编码 映射
print("标签映射到编码:", lable_to_index)  # 字典无序 不可索引
index_to_lable = dict((v, k) for k, v in lable_to_index.items())  # 字典的迭代
print("编码映射到标签:", index_to_lable)
all_labels = [lable_to_index.get(name) for name in all_label_names]
print("前三个对应标签:", all_labels[:3])

# 打乱数据集
np.random.seed(200)  # 保证使用同样的乱序方法  图片和标签是一一对应的关系
random_index = np.random.permutation(len(imgs_path))
imgs_path = np.array(imgs_path)[random_index]
all_labels = np.array(all_labels)[random_index]
# 划分训练集
i = int(len(imgs_path) * 0.8)
train_path = imgs_path[:i]
train_labels = all_labels[:i]
test_path = imgs_path[i:]
test_labels = all_labels[i:]
train_ds = tf.data.Dataset.from_tensor_slices((train_path, train_labels))  # 训练集train dataset
test_ds = tf.data.Dataset.from_tensor_slices((test_path, test_labels))  # 双重括号 里面作为一个整体


# 对图片进行处理
def load_img(path, label):  # 定义一个函数  load_img是函数名 小括号里是参数
    # image = cv2.imread(str(path))
    # image = cv2.resize(image, dsize=(256, 256))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, [256, 256])
    # image = tf.image.rgb_to_grayscale(image)
    image = tf.cast(image, tf.float32)
    image = image / 255

    return image, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.map(load_img, num_parallel_calls=AUTOTUNE)  # 第二个参数为了充分利用cpu
test_ds = test_ds.map(load_img, num_parallel_calls=AUTOTUNE)  # map方法是将每个元素都使用一个函数得出值
print(train_ds)

BATCH_SIZE = 32
train_ds = train_ds.repeat().shuffle(100).batch(BATCH_SIZE)
print(train_ds)  # 多出来一个维度

test_ds = test_ds.batch(BATCH_SIZE)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=(256, 256, 3), activation="relu", padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.GlobalAveragePooling2D())

model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(12, activation='softmax'))

print(model.summary())
model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["acc"])

train_count = len(train_path)
test_count = len(test_path)
step_per_epoch = train_count // BATCH_SIZE
validation_step = test_count // BATCH_SIZE

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc',
                                                 factor=0.1,
                                                 patience=3,
                                                 verbose=1,
                                                 mode='auto',
                                                 epsilon=0.0001,
                                                 cooldown=0,
                                                 min_lr=0)

history = model.fit(train_ds, epochs=100,
                    steps_per_epoch=step_per_epoch,
                    validation_data=test_ds,
                    validation_steps=validation_step,
                    callbacks=[reduce_lr])
print(history)
model.save(r'E:\model change 1\model.h5')

plt.rcParams["font.sans-serif"] = "SimHei"
plt.rcParams['axes.unicode_minus'] = False
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylim(0, 1)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.savefig(r'E:\model change 1\picture_1.jpg')
plt.show()
