# Live Twitter Sentiment Analysis
Coursework project for CSE 573 @ ASU to analyze live sentiment of Twitter tweets
![ui](https://user-images.githubusercontent.com/18646185/230970017-ff4f13a6-01f3-4f0f-9457-6c09566f91f4.png)

### Application Architecture
<img width="1009" alt="Screen Shot 2023-04-10 at 11 35 26 AM" src="https://user-images.githubusercontent.com/18646185/230970418-60a2167b-6130-4721-bd75-a18102519494.png">

### Multi Layer Perceptron Architecture
<img width="848" alt="Screen Shot 2023-04-10 at 11 35 50 AM" src="https://user-images.githubusercontent.com/18646185/230970504-de1f04d7-c7d5-4c31-bb35-bf14e9f70606.png">

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
