# Live Twitter Sentiment Analysis
Coursework project for CSE 573 @ ASU to analyze live sentiment of Twitter tweets
![Screenshot (64)](https://user-images.githubusercontent.com/18646185/162799999-9e196788-0b3a-40cf-a517-f20167d3121e.png)

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
