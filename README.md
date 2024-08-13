# Logistic-regression
Предлагается реализовать модель логистической регрессии о обучить ее на датасете изображений. Изображения двух типов - на одних присутствует коробка, на других нет

## Загрузка данных
``` python
def read_files(path: str, ans: int, target_dim: tuple = (256, 256)):
    files = os.listdir(path)
    X = None
    for i, name in enumerate(files):
        img = cv2.imread(path + '/' + name, 0) # 0 means black-white picture
        if img.shape != 0:
            img = cv2.resize(img, (256, 256))
            vect = img.reshape(1, 256 ** 2) / 255.

            X = vect if (X is None) else np.vstack((X, vect))
        print(f"{i}/{len(files)}")
    print()
    y = np.ones((len(X),1)) * ans
    return X, y
```
Для того чтобы загрузить данные в нейросеть необходимо должным образом открыть и преобразовать в вектор с числами. Для этого воспользуемся функцией read_files().

Этот парсер делает следующие вещи:

- открывает файл картинки с диска (с помощью библиотеки opencv),
- проверяет, что картинка действительно открылась и сейчас является матрицей (np.array),
- преобразует матрицу в вектор (путем записи всех столбцов друг под другом),
- возвращает массив из векторов, в которых хранятся картинки, и лейбл, соответствующий каждой картинке.

  ## Forward propogation
  для i-ого объекта:
``` python
  z = np.dot(w.T, X) + b
  A = sigmoid(z)
```

## Back propogation
Функция потерь:
```python
cost = (-np.sum(Y*np.log(A)+(1-Y)*np.log(1-A)))/m
```
подсчитываем градиент функции ошибки:
```python
dw = (np.dot(X, (A-Y).T))/m
db = np.average(A-Y)
```

## Обновление параметров
обновляем параметры в функции `optimize()`
```python
w = w - learning_rate*dw
b = b - learning_rate*db
```


## График зависимости значения функции потерь от количества итераций
(при разных значениях параметра `learning_rate`)
![image](https://github.com/user-attachments/assets/76799bc9-9daf-48b7-ac80-bf19b0112683)

## Точность модели
Если `Y_prediction_train` и `y_train` совпали, то это означает, что наше предположение о том, есть ли коробка, совпало с правильным ответом. Далее просто посчитаем среднее значение 
```python
print("train accuracy: {} %".format(100-np.mean(np.abs(Y_prediction_train - y_train)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - y_test)) * 100))
```
![photo_5321071056900972576_y](https://github.com/user-attachments/assets/5b568ec6-83b2-4597-be16-a9ee2c63a5fa)
