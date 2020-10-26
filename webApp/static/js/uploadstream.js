var U = function() {
	this.getScrollTop = function() {
		var scrollTop = 0;
		if(document.documentElement && document.documentElement.scrollTop) {
			scrollTop = document.documentElement.scrollTop;
		} else if(document.body) {
			scrollTop = document.body.scrollTop;
		}
		return scrollTop;
	}
	this.ele = function(text) {
		if(text != null & text != "") {
			if(text.length > 1) {
				if(/^#[a-zA-Z0-9_]*$/.test(text)) {
					return document.getElementById(text.substring(1, text.length));
				} else if(/^.[a-zA-Z0-9_]*$/.test(text)) {
					return document.getElementsByClassName(text.substring(1, text.length));
				} else {
					return document.getElementsByTagName(text);
				}
			}
		}
		return null;
	}
	this.inputTextLength = function(element, textLength) {
		var inputText = element.value;
		var inputTextLength = inputText.length;
		if(inputTextLength >= textLength) {
			element.value = inputText.substring(0, textLength);
		}
	}
	this.InnerText = function(ele, text) {
		var inner = ele.innerText;
		if(inner != null) {
			ele.innerText = text;
		} else {
			ele.textContent = text;
		}
	}
	this.forEach = function(array, fun) {
		var dataType = typeof array;
		if(dataType == "object") {
			for(var i = 0; i < array.length; i++) {
				fun({
					index: i,
					data: array[i]
				});
			}
		} else if(dataType == "number") {
			for(var i = 0; i < array; i++) {
				fun({
					index: i,
					data: null
				});
			}
		}
	}
	this.css = function(el, cssStyles) {
		if(typeof el == "object") {
			var css = {};
			var oldCss = el.getAttribute("style");
			if(oldCss != null) {
				var oldCsss = oldCss.split(";");
				this.forEach(oldCsss, function(e) {
					var keyAndValue = e.data;
					if(keyAndValue != "") {
						var k = keyAndValue.substring(0, keyAndValue.indexOf(":"));
						var v = keyAndValue.substring(keyAndValue.indexOf(":") + 1, keyAndValue.length);
						css[k] = v.trim();
					}
				})
			}
			for(cssStyle in cssStyles) {
				css[cssStyle] = cssStyles[cssStyle];
			}
			if(JSON.stringify(css) != "{}") {
				oldCss = "";
				for(c in css) {
					oldCss += c + ":" + css[c] + ";";
				}
			}
			el.setAttribute("style", oldCss);
		}
	}
	this.createElement = function(eleName, attrs, write, parseEle) {
		if(typeof eleName == "string") {
			var ele = document.createElement(eleName);
			if(typeof attrs == "object") {
				for(attr in attrs) {
					ele.setAttribute(attr, attrs[attr]);
				}
			}
			if(typeof write == "object") {
				for(text in write) {
					if(text == "text") {
						this.InnerText(ele, write[text]);
					} else if(text == "html") {
						ele.innerHTML = write[text];
					} else if(text == "value") {
						ele.value = write[text];
					}
				}
			}
			if(typeof parseEle == "object") {
				parseEle.appendChild(ele);
			}
			return ele;
		}
	}
	this.splitArray = function(array, arrayLength) {
		var VarTempArray = [];
		var temp = [];
		forEach(array, function(e) {
			temp.push(e.data);
			if(e.index % arrayLength == (arrayLength - 1) || e.index == (array.length - 1)) {
				VarTempArray.push(temp);
				temp = [];
			}
		});
		return VarTempArray;
	}
	this.format = function(timeStamp, timeFormat) {
		var date = new Date(timeStamp);
		var y = date.getFullYear();
		var M = (date.getMonth() + 1 < 10 ? '0' + (date.getMonth() + 1) : date.getMonth() + 1);
		var d = date.getDate() < 10 ? '0' + date.getDate() + '' : date.getDate() + '';
		var h = date.getHours() < 10 ? '0' + date.getHours() : date.getHours();
		var m = date.getMinutes() < 10 ? '0' + date.getMinutes() : date.getMinutes();
		var s = date.getSeconds() < 10 ? '0' + date.getSeconds() : date.getSeconds();
		if(timeFormat.indexOf("yyyy") != -1 || timeFormat.indexOf("YYYY") != -1) {
			timeFormat = timeFormat.replace(new RegExp(/yyyy|YYYY/g), y);
		}
		if(timeFormat.indexOf("MM") != -1) {
			timeFormat = timeFormat.replace(new RegExp(/MM/g), M);
		}
		if(timeFormat.indexOf("dd") != -1 || timeFormat.indexOf("DD") != -1) {
			timeFormat = timeFormat.replace(new RegExp(/dd|DD/g), d);
		}
		if(timeFormat.indexOf("hh") != -1 || timeFormat.indexOf("HH") != -1) {
			timeFormat = timeFormat.replace(new RegExp(/hh|HH/g), h);
		}
		if(timeFormat.indexOf("mm") != -1) {
			timeFormat = timeFormat.replace(new RegExp(/mm/g), m);
		}
		if(timeFormat.indexOf("ss") != -1 || timeFormat.indexOf("SS") != -1) {
			timeFormat = timeFormat.replace(new RegExp(/ss|SS/g), s);
		}
		return timeFormat;
	}
}

var ajax=function(method,url,data,callback,type){
	var xhr;
	if (window.XMLHttpRequest){
	  xhr=new XMLHttpRequest();
	}else{
	  xhr=new ActiveXObject("Microsoft.XMLHTTP");
	}
	xhr.onreadystatechange = function (){
		if(xhr.status==200 && xhr.readyState==4){
			if(type=='json'){
				var res = JSON.parse(xhr.responseText);
			}else if(type=='xml'){
				var res = responseXML;
			}else{
				var res = xhr.responseText;
			}
			callback(res);
		}
	};

	if(method != "formData"){
		var param = '';
		if(JSON.stringify(data) != '{}'){
			url += '?';
			for(var i in data){
				param += i+'='+data[i]+'&';
			}
			param = param.slice(0,param.length-1);
		}
		if(method == "get"){
			url = url+param;
		}
		xhr.open(method,url,true);
	} else {
		xhr.open("post",url,true);
	}

	if(method == "post"){
		xhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
		xhr.send(param);
	}else if(method == "formData"){
		xhr.send(data);
	}else{
		xhr.send(null);
	}
}

var media = function(obj) {
	this.video;
	this.uploadURI;
	this.uploadParams;
	this.recorderFile;
	this.mediaStream;
	this.mediaRecorder;
	this.recorderState;

	var MediaUtils = {
		getUserMedia: function(videoEnable, audioEnable, callback) {
			navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia || window.getUserMedia;
			var constraints = {
				video: videoEnable,
				audio: audioEnable
			};
			if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
				navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
					callback(false, stream);
				})['catch'](function(err) {
					callback(err);
				});
			} else if(navigator.getUserMedia) {
				navigator.getUserMedia(constraints, function(stream) {
					callback(false, stream);
				}, function(err) {
					callback(err);
				});
			} else {
				callback(new Error('Not support userMedia'));
			}
		},
		// close stream
		closeStream: function(stream) {
			if(typeof stream.stop === 'function') {
				stream.stop();
			} else {
				var trackList = [stream.getAudioTracks(), stream.getVideoTracks()];
				for(var i = 0; i < trackList.length; i++) {
					var tracks = trackList[i];
					if(tracks && tracks.length > 0) {
						for(var j = 0; j < tracks.length; j++) {
							var track = tracks[j];
							if(typeof track.stop === 'function') {
								track.stop();
							}
						}
					}
				}
			}
		}
	};

	//initial
	this.init = function() {
		var u = new U();
		this.uploadURI = obj["uploadURI"];
		this.uploadParams = obj["params"];
		this.video = u.ele(obj["video"]);
		if(this.video != null) {
			this.recorderState = false;
			MediaUtils.getUserMedia(true, false, (err, stream) => {
				if(err) {
					throw err;
				} else {
					this.mediaRecorder = new MediaRecorder(stream);
					this.mediaStream = stream;
					var chunks = [];
					this.video.srcObject = stream;
					this.video.play();
					this.mediaRecorder.ondataavailable = function(e) {
						this.blobs.push(e.data);
						chunks.push(e.data);
					};
					var _this = this;
					this.mediaRecorder.blobs = [];
					this.mediaRecorder.onstop = function(e) {
						_this.recorderFile = new Blob(chunks, {
							'type': _this.mediaRecorder.mimeType
						});
						chunks = [];
						_this.uploadFile(_this.recorderFile);
					};
				}
			});
		}
	}

	//record and upload to server
	this.uploadFile = function(recorderFile) {
		var file = new File([recorderFile], 'msr-' + (new Date).toISOString().replace(/:|\./g, '-') + '.mp4', {
			type: 'video/mp4'
		});
		var url = this.uploadURI;
		if(url != undefined & url != null & url != "") {
			if(confirm("record success,do you want to upload？")) {
				$('.loading').show();
				var formData = new FormData();
				formData.append("uploadFile", file);
				var params = this.uploadParams;
				if(params != undefined & params != null & typeof params == "object") {
					for(param in params) {
						formData.append(param, params[params]);
					}
				}
				ajax("formData", url, formData, function(response) {
					$('.loading').hide();
					console.log(response);

                    var processedFileName = JSON.parse(response).filename;
					$("#downloadbtn").attr("href","/download/"+processedFileName)
                    $("#processresultdiv").show();
				},'text');
				this.mediaRecorder = null;
				this.mediaStream = null;
				this.recorderFile = null;
			}
		}
	}

	this.uploadImageFile = function(image, username) {
		var formData = new FormData();
		formData.append("imageBase64", image);
		formData.append("username", username);
		$('.loading').show();
		ajax("formData", '/uploadImageBase64', formData, function(response) {
			$('.loading').hide();
            console.log(response)

            var processedFileName = JSON.parse(response).filename;
			$("#uploadFile_processed").attr("src","/static/processed/"+processedFileName)
			$("#processresultdiv").show();
		},'text');
	}

	//capture,return url
	this.screenshot=function(){
		var canvas=document.createElement("canvas");
		canvas.width=this.video.videoWidth;
		canvas.height=this.video.videoHeight;
		canvas.getContext('2d').drawImage(this.video, 0, 0, canvas.width, canvas.height);
		return canvas.toDataURL("image/png");
	}

	//start record
	this.startRecorder = function() {
		if(this.mediaRecorder != undefined & this.mediaRecorder != null) {
			this.recorderState = true;
			this.mediaRecorder.start();
		}
	}

	//end record
	this.stopRecorder = function() {
		if(this.mediaRecorder != undefined & this.mediaRecorder != null) {
			setTimeout(() => {
				this.recorderState = false;
				this.mediaRecorder.stop();
				MediaUtils.closeStream(this.mediaStream);
			}, 2000);
		}
	}

	//close camera
	this.closeMedia = function() {
		if(this.mediaStream != undefined & this.mediaStream != null) {
			MediaUtils.closeStream(this.mediaStream);
		}
	}
}