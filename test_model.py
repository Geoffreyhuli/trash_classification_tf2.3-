import tensorflow as tf
import os
import numpy as np
from PIL import Image
import shutil
import cv2
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize



# 可以显示中文
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']

# 垃圾类名
class_names = ['其他垃圾_PE塑料袋', '其他垃圾_U型回形针', '其他垃圾_一次性杯子', '其他垃圾_一次性棉签', '其他垃圾_串串竹签', '其他垃圾_便利贴', '其他垃圾_创可贴',
               '其他垃圾_厨房手套', '其他垃圾_口罩', '其他垃圾_唱片', '其他垃圾_图钉', '其他垃圾_大龙虾头', '其他垃圾_奶茶杯', '其他垃圾_干果壳', '其他垃圾_干燥剂',
               '其他垃圾_打泡网', '其他垃圾_打火机', '其他垃圾_放大镜', '其他垃圾_毛巾', '其他垃圾_涂改带', '其他垃圾_湿纸巾', '其他垃圾_烟蒂', '其他垃圾_牙刷', '其他垃圾_百洁布',
               '其他垃圾_眼镜', '其他垃圾_票据', '其他垃圾_空调滤芯', '其他垃圾_笔及笔芯', '其他垃圾_纸巾', '其他垃圾_胶带', '其他垃圾_胶水废包装', '其他垃圾_苍蝇拍', '其他垃圾_茶壶碎片', '其他垃圾_餐盒',
               '其他垃圾_验孕棒', '其他垃圾_鸡毛掸', '厨余垃圾_八宝粥', '厨余垃圾_冰糖葫芦', '厨余垃圾_咖啡渣', '厨余垃圾_哈密瓜', '厨余垃圾_圣女果', '厨余垃圾_巴旦木', '厨余垃圾_开心果', '厨余垃圾_普通面包',
               '厨余垃圾_板栗', '厨余垃圾_果冻', '厨余垃圾_核桃', '厨余垃圾_梨', '厨余垃圾_橙子', '厨余垃圾_残渣剩饭', '厨余垃圾_汉堡', '厨余垃圾_火龙果', '厨余垃圾_炸鸡', '厨余垃圾_烤鸡烤鸭', '厨余垃圾_牛肉干',
               '厨余垃圾_瓜子', '厨余垃圾_甘蔗', '厨余垃圾_生肉', '厨余垃圾_番茄', '厨余垃圾_白菜', '厨余垃圾_白萝卜', '厨余垃圾_粉条', '厨余垃圾_糕点', '厨余垃圾_红豆', '厨余垃圾_肠(火腿)', '厨余垃圾_胡萝卜',
               '厨余垃圾_花生皮', '厨余垃圾_苹果', '厨余垃圾_茶叶', '厨余垃圾_草莓', '厨余垃圾_荷包蛋', '厨余垃圾_菠萝', '厨余垃圾_菠萝包', '厨余垃圾_菠萝蜜', '厨余垃圾_蒜', '厨余垃圾_薯条', '厨余垃圾_蘑菇',
               '厨余垃圾_蚕豆', '厨余垃圾_蛋', '厨余垃圾_蛋挞', '厨余垃圾_西瓜皮', '厨余垃圾_贝果', '厨余垃圾_辣椒', '厨余垃圾_陈皮', '厨余垃圾_青菜', '厨余垃圾_饼干', '厨余垃圾_香蕉皮', '厨余垃圾_骨肉相连',
               '厨余垃圾_鸡翅', '可回收物_乒乓球拍', '可回收物_书', '可回收物_保温杯', '可回收物_保鲜盒', '可回收物_信封', '可回收物_充电头', '可回收物_充电宝', '可回收物_充电线', '可回收物_八宝粥罐', '可回收物_刀',
               '可回收物_剃须刀片', '可回收物_剪刀', '可回收物_勺子', '可回收物_单肩包手提包', '可回收物_卡', '可回收物_叉子', '可回收物_变形玩具', '可回收物_台历', '可回收物_台灯', '可回收物_吹风机', '可回收物_呼啦圈',
               '可回收物_地球仪', '可回收物_地铁票', '可回收物_垫子', '可回收物_塑料瓶', '可回收物_塑料盆', '可回收物_奶盒', '可回收物_奶粉罐', '可回收物_奶粉罐铝盖', '可回收物_尺子', '可回收物_帽子', '可回收物_废弃扩声器',
               '可回收物_手提包', '可回收物_手机', '可回收物_手电筒', '可回收物_手链', '可回收物_打印机墨盒', '可回收物_打气筒', '可回收物_护肤品空瓶', '可回收物_报纸', '可回收物_拖鞋', '可回收物_插线板', '可回收物_搓衣板',
               '可回收物_收音机', '可回收物_放大镜', '可回收物_易拉罐', '可回收物_暖宝宝', '可回收物_望远镜', '可回收物_木制切菜板', '可回收物_木制玩具', '可回收物_木质梳子', '可回收物_木质锅铲', '可回收物_枕头', '可回收物_档案袋',
               '可回收物_水杯', '可回收物_泡沫盒子', '可回收物_灯罩', '可回收物_烟灰缸', '可回收物_烧水壶', '可回收物_热水瓶', '可回收物_玩偶', '可回收物_玻璃器皿', '可回收物_玻璃壶', '可回收物_玻璃球', '可回收物_电动剃须刀',
               '可回收物_电动卷发棒', '可回收物_电动牙刷', '可回收物_电熨斗', '可回收物_电视遥控器', '可回收物_电路板', '可回收物_登机牌', '可回收物_盘子', '可回收物_碗', '可回收物_空气加湿器', '可回收物_空调遥控器', '可回收物_纸牌',
               '可回收物_纸箱', '可回收物_罐头瓶', '可回收物_网卡', '可回收物_耳套', '可回收物_耳机', '可回收物_耳钉耳环', '可回收物_芭比娃娃', '可回收物_茶叶罐', '可回收物_蛋糕盒', '可回收物_螺丝刀', '可回收物_衣架', '可回收物_袜子',
               '可回收物_裤子', '可回收物_计算器', '可回收物_订书机', '可回收物_话筒', '可回收物_购物纸袋', '可回收物_路由器', '可回收物_车钥匙', '可回收物_量杯', '可回收物_钉子', '可回收物_钟表', '可回收物_钢丝球', '可回收物_锅',
               '可回收物_锅盖', '可回收物_键盘', '可回收物_镊子', '可回收物_鞋', '可回收物_餐垫', '可回收物_鼠标', '有害垃圾_LED灯泡', '有害垃圾_保健品瓶', '有害垃圾_口服液瓶', '有害垃圾_指甲油', '有害垃圾_杀虫剂', '有害垃圾_温度计',
               '有害垃圾_滴眼液瓶', '有害垃圾_玻璃灯管', '有害垃圾_电池', '有害垃圾_电池板', '有害垃圾_碘伏空瓶', '有害垃圾_红花油', '有害垃圾_纽扣电池', '有害垃圾_胶水', '有害垃圾_药品包装', '有害垃圾_药片', '有害垃圾_药膏', '有害垃圾_蓄电池', '有害垃圾_血压计']


# 大类名
major_class_names = ['厨余垃圾', '可回收物', '其他垃圾', '有害垃圾']

# 数据加载，按照8:2的比例加载垃圾数据
def data_load(data_dir, img_height, img_width, batch_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names
    return train_ds, val_ds, class_names

# 测试mobilenet的准确率
def test_mobilenet(train_ds, val_ds, class_names):
    model = tf.keras.models.load_model("models/mobilenet.h5")
    loss, accuracy = model.evaluate(val_ds)
    print('Mobilenet test accuracy :', accuracy)

    # 生成 ROC 曲线
    y_true = []
    y_scores = []
    for images, labels in val_ds:
        predictions = model.predict(images)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_scores.extend(predictions)

    y_true = label_binarize(y_true, classes=range(len(class_names)))
    y_scores = np.array(y_scores)

    # 计算每个大类的 ROC 曲线和 AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i, major_class_name in enumerate(major_class_names):
        # 提取每个 class_name 的前缀进行匹配
        idx = [index for index, name in enumerate(class_names) if name.split('_')[0] == major_class_name]
        if len(idx) > 0:
            idx = idx[0]  # 如果有多个匹配项，这里只取第一个（可以根据需求调整处理方式）
        else:
            raise ValueError(f"大类 '{major_class_name}' 不存在于 class_names 中")

        fpr[i], tpr[i], _ = roc_curve(y_true[:, idx], y_scores[:, idx])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 绘制 ROC 曲线
    plt.figure()
    for i, class_name in enumerate(major_class_names):
        plt.plot(fpr[i], tpr[i], label='{} (AUC = {:.2f})'.format(class_name, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真正率')
    plt.title('ROC 曲线')
    plt.legend(loc="lower right")
    plt.savefig("F:\\trash_classification_tf2.3\\results\\roc_curve.png")  # 保存图片
    plt.close()

# 注：绘制热力图这段的逻辑
def draw_heatmap(folder_name, model, class_names, major_class_names):
    trash_names = major_class_names
    real_label = []
    pre_label = []
    images_path = []
    folders = os.listdir(folder_name)

    for folder in folders:
        folder_path = os.path.join(folder_name, folder)
        images = os.listdir(folder_path)
        for img in images:
            xxx = folder.split("_")[0]
            x_idx = trash_names.index(xxx)
            img_path = os.path.join(folder_path, img)
            real_label.append(x_idx)
            images_path.append(img_path)

    for ii, i_path in enumerate(images_path):
        print("{}/{}".format(ii, len(images_path) - 1))
        shutil.copy(i_path, "images/t1.jpg")
        src_i = cv2.imread("images/t1.jpg")
        src_r = cv2.resize(src_i, (224, 224))
        cv2.imwrite("images/t2.jpg", src_r)
        img = Image.open("images/t2.jpg")
        img = np.asarray(img)
        outputs = model.predict(img.reshape(1, 224, 224, 3))
        result_index = int(np.argmax(outputs))
        result = class_names[result_index]
        names = result.split("_")
        xxx = names[0]
        x_idx = trash_names.index(xxx)
        pre_label.append(x_idx)

    print(len("pre:{}".format(len(pre_label))))
    print(len("real:{}".format(len(real_label))))
    print(pre_label)
    print(real_label)
    # 先保存为pickle文件
    a_dict = {"pre_label": pre_label, "real_label": real_label}
    with open('results/pickle_result.pickle', 'wb') as file:
        pickle.dump(a_dict, file)

# 加载pkl文件
def load_pickle(filename="results/pickle_result.pickle"):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    pre_label = data['pre_label']
    real_label = data['real_label']
    print(len(pre_label))
    print(len(real_label))
    heatmap = np.zeros((4, 4))
    for r, p in zip(real_label, pre_label):
        heatmap[r][p] += 1
    print(heatmap)
    result = []
    for row in heatmap:
        row = row / np.sum(row)
        result.append(row)
    return np.array(result)

# 得到热力图
def get_heat_map(array_numpy=np.random.rand(16).reshape(4, 4)):
    trash_names = major_class_names
    plt.xticks(np.arange(len(trash_names)), trash_names)
    plt.yticks(np.arange(len(trash_names)), trash_names)
    plt.imshow(array_numpy, cmap=plt.cm.hot, vmin=0, vmax=1)
    plt.title('分类热力图')
    plt.colorbar()
    plt.savefig("F:\\trash_classification_tf2.3\\results\\heatmap.png")  # 保存图片
    plt.close()

if __name__ == '__main__':
    train_ds, val_ds, class_names = data_load("F://trash_jpg", 224, 224, 80)
    test_mobilenet(train_ds, val_ds, class_names)
    model = tf.keras.models.load_model("models/mobilenet.h5")
    draw_heatmap("F://trash_jpg", model, class_names, major_class_names)
    heatmap_data = load_pickle()
    get_heat_map(heatmap_data)