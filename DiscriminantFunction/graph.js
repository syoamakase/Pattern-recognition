google.load("visualization", "1", { packages: ["corechart"] });
google.setOnLoadCallback(drawChart);
graph_data = [
  ['w0', 'w1'],
  [0, 0],
  [15,15]
];
var data;
var line = 0;
function drawChart() {
  data = google.visualization.arrayToDataTable(graph_data);
  var options = {
    title: 'Perceptron',
    hAxis: { title: 'w1', minValue: 0, maxValue: 15 },
    vAxis: { title: 'w0', minValue: 0, maxValue: 15 },
    legend: 'none',
  };
  var chart = new google.visualization.LineChart(document.getElementById('chart_div'));
  chart.draw(data, options);
  }
  function changeGraph(vec){
    var newvalue = data.getValue(1,0);
	//console.log(newvalue);
	var xmlHttpRequest = new XMLHttpRequest();
    xmlHttpRequest.open( 'GET', './data.json', true );
    xmlHttpRequest.responseType = 'json';
	xmlHttpRequest.onload = function(){
	  console.log(xmlHttpRequest.responseText);
	  var myData = JSON.parse(this.responseText);
	}
    xmlHttpRequest.send( null );
  }
google.setOnLoadCallback(drawChart);
