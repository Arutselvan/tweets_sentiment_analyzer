import './App.css';
import React, {useEffect, useRef } from 'react';
import useState from 'react-usestateref';

import 'bootstrap/dist/css/bootstrap.min.css';
import {Container, Badge, Row, Col, Navbar, Form, Button, FormControl, Card, Accordion} from 'react-bootstrap';

import Chart from "react-apexcharts";

import ReactWordcloud from 'react-wordcloud';

const primary = "#0d6efd";
const success = "#198754";
const danger = "#dc3545";

const pieOptions = {
  options: {
    chart: {
      width: 380,
      type: 'donut',
    },
    dataLabels: {
      enabled: false
    },
    labels: ['Positive', 'Negative', 'Neutral'],
    colors: [success, danger, primary],
    responsive: [{
      breakpoint: 480,
      options: {
        chart: {
          width: 200
        },
        legend: {
          show: false
        }
      }
    }],
    legend: {
      position: 'right',
      offsetY: 0,
      height: 230,
    }
  }
}

const chartOptions = {
  options: {
    chart: {
      id: 'realtime',
      height: 350,
      type: 'line',
      animations: {
        enabled: true,
        easing: 'linear',
        dynamicAnimation: {
          speed: 200
        }
      },
      toolbar: {
        show: false
      },
      zoom: {
        enabled: false
      }
    },
    dataLabels: {
      enabled: false
    },
    stroke: {
      curve: 'smooth'
    },
    markers: {
      size: 0
    },
    xaxis: {
      show: false
    },
    yaxis: {
      max: 1,
      min: -1
    },
    legend: {
      show: false
    },
  }
};

const cloudOptions = {
  colors: ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
  enableTooltip: true,
  deterministic: false,
  fontFamily: "impact",
  fontSizes: [5, 60],
  fontStyle: "normal",
  fontWeight: "normal",
  padding: 1,

  transitionDuration: 0
}

function App() {
  const [sentiment, setSentiment, sentiRef] = useState([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]);
  const [rolling, setRolling, rollingRef] = useState([]);
  const [color, setColor]  = useState(primary);
  const [rollingtotal, setRollingtotal, rollingtotalRef] = useState(0);
  const [total, setTotal, totalRef] = useState(0);
  const [poscount, setPoscount] = useState(0);
  const [cloud, setcloud, cloudRef] = useState([]);
  const [negcount, setNegcount] = useState(0);
  const [sentimentscore, setSentimentscore, scoreRef] = useState(0*1.0);
  const webSocket = useRef(null);
  const [tweets, setTweets, tweetsRef] = useState([])
  const [keywords, setKeywords, keywordsRef] = useState([])

  useEffect(() => {
    webSocket.current = new WebSocket("ws://localhost:8000/ws/stream/");
    webSocket.current.onmessage = (message) => {
      let obj = JSON.parse(message.data);
      setTotal(prev => prev + 1);
      if(rollingtotalRef.current < 15){
        setRollingtotal(prev => prev + 1)
      }
      let temp_score = 0;
      let temp_rolling = rollingRef.current;
      let temp_tweets = tweetsRef.current;
      let temp_keywords = keywordsRef.current;
      if(obj.sentiment==="POSITIVE"){
        setPoscount(prev => prev + 1);
        // temp_score = (scoreRef.current*(totalRef.current-1)*1.0 + 1)/totalRef.current;
        if(totalRef.current>15){
          temp_rolling.shift()
        }
        temp_rolling.push(1)
        temp_score = (temp_rolling.reduce((a, b) => a + b, 0)/rollingtotalRef.current)*1.0
        setRolling(temp_rolling)
        setSentimentscore(temp_score.toFixed(2));
      }
      else if(obj.sentiment==="NEGATIVE"){
        setNegcount(prev => prev + 1);
        if(totalRef.current>15){
          temp_rolling.shift()
        }
        temp_rolling.push(-1)
        temp_score = (temp_rolling.reduce((a, b) => a + b, 0)/rollingtotalRef.current)*1.0
        setRolling(temp_rolling)
        setSentimentscore(temp_score.toFixed(2));
      }
      else{
        if(totalRef.current>15){
          temp_rolling.shift()
        }
        temp_rolling.push(0)
        temp_score = (temp_rolling.reduce((a, b) => a + b, 0)/rollingtotalRef.current)*1.0
        setRolling(temp_rolling)
        setSentimentscore(temp_score.toFixed(2));
      }

      if(totalRef.current>15){
        temp_tweets.pop()
        temp_keywords.pop()
      }

      temp_tweets.unshift({tweet:obj.tweet, score: obj.weight, sentiment: obj.sentiment})
      temp_keywords.unshift(obj.word_map)

      setTweets(temp_tweets)
      setKeywords(temp_keywords)


      if(scoreRef.current===0){
        setColor(primary);
      }
      else if(scoreRef.current>0){
        setColor(success)
      }
      else{
        setColor(danger)
      }

      let temp = sentiRef.current;

      if(temp.length>=15){
        temp.shift()
      }
      temp.push(scoreRef.current)
      setSentiment(temp)

      let temp_cloud = {}

      cloudRef.current.forEach(element => {
        temp_cloud[element.text] = element.value;
      });

      Object.entries(obj.word_map).forEach(
        ([key, value]) => {temp_cloud[key] = (temp_cloud[key]+1) || 1 ;}
      );

      let list_cloud = []

      Object.entries(temp_cloud).forEach(
        ([key, value]) => {list_cloud.push({text:key, value:value})}
      );

      setcloud(list_cloud)


    };
    return () => webSocket.current.close();
  }, []);

  const handleSubmit = (event) => {
    event.preventDefault();
    let data = new FormData(event.target);

    let topic = data.get('topic'); // reference by form input's `name` tag

    webSocket.current.send(topic)
    setTotal(0)
    setPoscount(0)
    setSentimentscore(0)
    setNegcount(0)
    setRolling([])
    setRollingtotal(0)
    setColor(primary)
    setSentiment([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    setcloud([])
    setTweets([])
    setKeywords([])

  };

  const colorMap = {
    1: success,
    0: primary,
    "-1": danger 
  }

  const sentiColor = {
    "POSITIVE": 'success',
    "NEGATIVE": 'danger',
    "NEUTRAL": 'primary'
  }

  const sentiColorVar = {
    "POSITIVE": success,
    "NEGATIVE": danger,
    "NEUTRAL": primary
  }

  const wordButtons = (words) => {
    let butts = []
    for(const prop in words){
      butts.push(<Button style={{'margin': `5px`}} variant={sentiColor[words[prop].sentiment]}>
      {prop} <Badge bg="secondary">{words[prop].weight.toFixed(2)}</Badge>
      <span className="visually-hidden">score</span>
    </Button>)
    }
    return butts;
  };

  const tweetBlocks = () => tweets.map((tweet,i) => 
    <Accordion.Item eventKey={i}>
      <Accordion.Header style={{'backgroundColor': sentiColorVar[tweet.sentiment]}}>{tweet.tweet}<Button style={{'margin': `2px`, 'fontSize': '10px'}} variant={sentiColor[tweet.sentiment]}>
      {tweet.sentiment} <Badge bg="secondary">{tweet.score.toFixed(2)}</Badge>
      <span className="visually-hidden">score</span>
    </Button></Accordion.Header>
      <Accordion.Body>
        {wordButtons(keywords[i])}
      </Accordion.Body>
    </Accordion.Item>
);

  return (
    <div className="App">
      <Navbar>
        <Container style={{'padding': '15px'}}>
          <Navbar.Brand href="#home">Live Twitter Sentiment Analysis</Navbar.Brand>
          <Navbar.Toggle />
          <Navbar.Collapse className="justify-content-end">
            <Navbar.Text>
              Developed for CSE 573 Spring 22 @ ASU by Group 4
            </Navbar.Text>
          </Navbar.Collapse>
        </Container>
      </Navbar>
      <Container>
      <Row style={{'marginBottom': '15px'}}>
          <Col>
          <Form onSubmit={handleSubmit} className="d-flex">
            <FormControl
              type="search"
              placeholder="Track topic"
              className="lg-1"
              aria-label="Track"
              name="topic"
            />
            <Button type="submit" style={{'marginLeft': '15px'}} variant="primary">Track</Button>
          </Form>
          </Col>
        </Row>
        <Row>
          <Col>
            <Card>
              <Card.Header style={{'backgroundColor': primary, 'color':'white'}}>Total Tweets Analyzed</Card.Header>
              <Card.Body style={{'fontSize': '2rem'}}>
                {total}
              </Card.Body>
            </Card>
          </Col>
          <Col>
            <Card>
              <Card.Header style={{'backgroundColor': success, 'color':'white'}}  >Positive Tweets</Card.Header>
              <Card.Body style={{'fontSize': '2rem'}}>
                {poscount}
              </Card.Body>
            </Card>
          </Col>
          <Col>
            <Card>
              <Card.Header style={{'backgroundColor': danger, 'color':'white'}} >Negative Tweets</Card.Header>
              <Card.Body style={{'fontSize': '2rem'}}>
                {negcount}
              </Card.Body>
            </Card>
          </Col>
          <Col>
            <Card style={{'backgroundColor': color, 'color':'white'}}>
              <Card.Header >Current Sentiment Score</Card.Header>
              <Card.Body style={{'fontSize': '2rem'}}>
                {sentimentscore}
              </Card.Body>
            </Card>
          </Col>
        </Row>
       
        <Row style={{'marginTop': '15px'}}>
          <Col sm={12} md={8}>
            <Card style={{'textAlign': 'left'}}>
            <Card.Header>Live Sentiment Trend</Card.Header>
            <Card.Body>
              <Chart
                options={chartOptions.options}
                series={[{
                  data: sentiment.slice()
                }]}
                height="605px"
              />
            </Card.Body>
            </Card>
          </Col>
          <Col sm={12} md={4}>
          <Card>
          <Card.Header>Sentiment Donut</Card.Header>
          <Card.Body>
          <Chart
                options={pieOptions.options}
                series={[poscount, negcount, total - (poscount+negcount)]}
                type="donut"
              />
            </Card.Body>
              </Card>
              <Card>
            <Card.Header>Word Cloud</Card.Header>
            <Card.Body>
            <ReactWordcloud words={cloud} options={cloudOptions}/>
              </Card.Body>
                </Card>
          </Col>
        </Row>
        <Row style={{'marginTop': '15px'}}>
        <Col sm={12}>
          <Card style={{'textAlign': 'left'}}>
            <Card.Header>Tweets, keywords and weights</Card.Header>
            <Card.Body>
                <Accordion>
                  {tweetBlocks()}
                </Accordion>
            </Card.Body>
          </Card>
        </Col>
        </Row>
      </Container>
    </div>
  );
}

export default App;
