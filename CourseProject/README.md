#Итоговый проект (пример) курса "Машинное обучение в бизнесе"


Стек:

ML: sklearn, pandas, numpy API: flask


Данные: с kaggle - https://www.kaggle.com/vstepanenko/disaster-tweets

Задача: предсказать, действительно ли твит является сообщением о бедствии


Испольюуемые признаки:

- text (text): текст твита
- keyword (text)
- location (text): какая-то информация о месте 

Преобразования признаков: tfidf, OHE

Использование:

### Клонируем репозиторий и создаем образ

```
$ git clone https://github.com/nnnedelkina/MLInBusiness.git
$ cd MLInBusiness/CourseProject
$ docker build -t nnn/course_project .
```

### Запускаем контейнер
```
docker run -p 8180:8180 -v <полный путь к директории для журнала, напримр /var/log>:/app/log nnn/course_project
```

### Посылаем post-запросы 

с полями text, keyword, location на http://localhost:8180/predict

Предупреждения: 

- оставлена модель LogisticRegression, с RandomForestClassifier f-score был повыше, но не хватало памяти на загрузку в тестовом огружении
- f-score был 0.52-0.55, желательно дальнейшее улучшение модели. 








