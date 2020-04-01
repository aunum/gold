package ui

import (
	"time"

	"github.com/GeertJohan/go.rice/embedded"
)

func init() {

	// define files
	file2 := &embedded.EmbeddedFile{
		Filename:    "dashboard.html",
		FileModTime: time.Unix(1585752721, 0),

		Content: string("<!doctype html>\n<html lang=\"en\">\n\t<head>\n\t\t<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, shrink-to-fit=no\">\n\t\t<link rel=\"icon\" href=\"https://avatars1.githubusercontent.com/u/17137938?s=400&v=4\">\n\t\t<link rel=\"stylesheet\" href=\"https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css\" integrity=\"sha384-Vkoo8x4CGsO3+Hhxv8T/Q5PaXtkKtu6ug5TOeNV6gBiFeWPGFN9MuhOf23Q9Ifjh\" crossorigin=\"anonymous\">\n\t</head>\n\t<body>\n\t<ul class=\"nav\">\n\t\t<a class=\"navbar-brand\" id=\"brand\" href=\"#\" style=\"padding-left: 20px;\"></a>\n\t</ul>\n\t<div class=\"text-center\">\n\t\t<img id=\"live\" style=\"display:none\" class=\"img-fluid\"/>\n\t</div>\n\n\t<div class=\"d-flex justify-content-around flex-wrap\" id=\"metrics\">\n\t</div>\n\n\t<script src=\"https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js\"></script>\n\t<script src=\"https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.6.0/Chart.bundle.js\"></script>\n\t<script src=\"https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js\" integrity=\"sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo\" crossorigin=\"anonymous\"></script>\n\t</body>\n\t<script>\n\t\tvar getInfo = function() {\n\t\t\t$.ajax({\n\t\t\t\turl: \"/info\",\n\t\t\t\tdataType: \"json\",\n\t\t\t\tsuccess: function(response) {\n\t\t\t\t\ttitle = response.name + \" Agent\"\n\t\t\t\t\tdocument.title = title;\n\t\t\t\t\tvar node = document.createTextNode(title);\n\t\t\t\t\tdocument.getElementById(\"brand\").appendChild(node);\n\t\t\t\t}\n\t\t\t});\n\t\t};\n\t\tgetInfo();\n\t</script>\n\t<script>\n\t\tvar client = new EventSource(\"/live\")\n\t\tclient.onmessage = function (msg) {\n\t\t\tvar metaData = \"data:image/jpeg;base64,\";\n\t\t\tlive = document.getElementById(\"live\")\n\t\t\tlive.style = \"height:400px;width:600px\";\n\t\t\tlive.src = metaData + msg.data;\n\t\t}\n\t</script>\n\t<script>\n\t\twindow.localStorage.setItem('values', []);\n\t\tfunction buildChart(value) {\n\t\t\tvar ctx_live = document.getElementById(value);\n\t\t\tvar chart = new Chart(ctx_live, {\n\t\t\t\ttype: 'line',\n\t\t\t\tdata: {\n\t\t\t\tlabels: [],\n\t\t\t\tdatasets: [{\n\t\t\t\t\tdata: [],\n\t\t\t\t\tborderWidth: 1,\n\t\t\t\t\tborderColor:'#00c0ef',\n\t\t\t\t\tfill: false,\n\t\t\t\t}]\n\t\t\t\t},\n\t\t\t\toptions: {\n\t\t\t\tresponsive: true,\n\t\t\t\ttitle: {\n\t\t\t\t\tdisplay: true,\n\t\t\t\t\ttext: value,\n\t\t\t\t},\n\t\t\t\tlegend: {\n\t\t\t\t\tdisplay: false\n\t\t\t\t},\n\t\t\t\tscales: {\n\t\t\t\t\tyAxes: [{\n\t\t\t\t\t\ttype: 'linear',\n\t\t\t\t\t\tticks: {\n\t\t\t\t\t\t\tbeginAtZero: true,\n\t\t\t\t\t}\n\t\t\t\t\t}],\n\t\t\t\t\txAxes: [{\n\t\t\t\t\t\ttype: 'linear',\n\t\t\t\t\t\tticks: {\n\t\t\t\t\t\tbeginAtZero: true,\n\t\t\t\t\t\t},\n\t\t\t\t\t\tscaleLabel: {\n\t\t\t\t\t\t\tdisplay: true,\n\t\t\t\t\t\t\tlabelString: 'Episode'\n\t\t\t\t\t\t}\n\t\t\t\t\t}]\n\t\t\t\t}\n\t\t\t\t}\n\t\t\t});\n\t\t\tvar getData = function() {\n\t\t\t\t$.ajax({\n\t\t\t\t\turl: \"/api/values/\"+value,\n\t\t\t\t\tdataType: \"json\",\n\t\t\t\t\tsuccess: function(response) {\n\t\t\t\t\t\tchart.data.datasets[0].data = response.xys;\n\t\t\t\t\t\tchart.options.scales.xAxes[0].scaleLabel.labelString = response.xLabel;\n\n\t\t\t\t\t\tchart.update();\n\t\t\t\t\t}\n\t\t\t\t});\n\t\t\t};\n\t\t\tgetData();\n\t\t\tsetInterval(getData, 1000);\n\t\t};\n\n\n\t\tvar getValues = function() {\n\t\t\t$.ajax({\n\t\t\t\turl: \"/api/values\",\n\t\t\t\tdataType: \"json\",\n\t\t\t\tsuccess: function(response) {\n\t\t\t\t\tvalues = window.localStorage.getItem('values');\n\t\t\t\t\twindow.localStorage.setItem('values', response);\n\t\t\t\t\tif(!response) {\n\t\t\t\t\t\treturn\n\t\t\t\t\t};\n\t\t\t\t\tif(!values) {\n\t\t\t\t\t\tvalues = []\n\t\t\t\t\t};\n\t\t\t\t\t\n\t\t\t\t\tnewValues = response.filter(e => !values.includes(e))\n\t\t\t\t\tif (newValues) {\n\t\t\t\t\t\tvar body = document.getElementById(\"metrics\");\n\t\t\t\t\t\tfor (value of newValues) {\n\t\t\t\t\t\t\tconsole.log('adding value: ' + value)\n\t\t\t\t\t\t\tvar div = document.createElement(\"div\");\n\t\t\t\t\t\t\tdiv.id = value + 'Holder';\n\t\t\t\t\t\t\tdiv.class = \"p-2\"\n\n\t\t\t\t\t\t\tvar canvas = document.createElement('canvas');\n\t\t\t\t\t\t\tcanvas.id = value;\n\t\t\t\t\t\t\tcanvas.width = 400;\n\t\t\t\t\t\t\tcanvas.height = 400;\n\n\t\t\t\t\t\t\tdiv.appendChild(canvas);\n\n\t\t\t\t\t\t\tbody.appendChild(div);\n\n\t\t\t\t\t\t\tbuildChart(value)\n\t\t\t\t\t\t};\n\t\t\t\t\t};\n\t\t\t\t}\n\t\t\t});\n\t\t}\n\t\tgetValues()\n\t\tsetInterval(getValues, 2000);\n\t</script>\n</html>"),
	}

	// define dirs
	dir1 := &embedded.EmbeddedDir{
		Filename:   "",
		DirModTime: time.Unix(1584726425, 0),
		ChildFiles: []*embedded.EmbeddedFile{
			file2, // "dashboard.html"

		},
	}

	// link ChildDirs
	dir1.ChildDirs = []*embedded.EmbeddedDir{}

	// register embeddedBox
	embedded.RegisterEmbeddedBox(`./static`, &embedded.EmbeddedBox{
		Name: `./static`,
		Time: time.Unix(1584726425, 0),
		Dirs: map[string]*embedded.EmbeddedDir{
			"": dir1,
		},
		Files: map[string]*embedded.EmbeddedFile{
			"dashboard.html": file2,
		},
	})
}
