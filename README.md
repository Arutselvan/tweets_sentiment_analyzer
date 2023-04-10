# Live Twitter Sentiment Analysis
Coursework project for CSE 573 @ ASU to analyze live sentiment of Twitter tweets
![ui](https://user-images.githubusercontent.com/18646185/230970017-ff4f13a6-01f3-4f0f-9457-6c09566f91f4.png)


### Run Backend
Before trying to start the Django server, make sure Redis server is running using the default port 6379
```
cd sentibackend
pip install -r requirements.txt
python manage.py runserver
```

### Run Frontend
```
cd sentiment_dashboard
npm install
npm start
```
