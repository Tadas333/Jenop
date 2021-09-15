<!DOCTYPE html>
<html>

<head>


  

<style>
h1 {
  color: blue;
  font-family: verdana;
  font-size: 200%;
  border: 2px solid powderblue;
  padding: 30px;
}
p {
  color: red;
  font-family: courier;
  font-size: 160%;
  position: relative;
  left: 40px;
}

.container {
  display: flex;
  justify-content: top;
  align-items: top;
  width: 750px; 
  height: 750px;
  }

   .container > div > video {
   width: 40px;
   height: 40px;
   }

}
a {
  margin: 0px 150px;
}

.button {
  border: none;
  color: white;
  padding: 10px 20px;
  text-align: center;
  margin: 0px 0px;
  text-decoration: none;
  display: inline-block;
  font-size: 16px;
  
  transition-duration: 0.4s;
  cursor: pointer;
  
}
.button1 {
  background-color: white; 
  color: black; 
  border: 2px solid #4CAF50;
}

.button1:hover {
  background-color: #4CAF50;
  color: white;
}


</style>
</head>

<body>



<h1>Camara 1</h1>
<p>Connected
<a href="http://192.168.1.128/10min_recs" target="_blank"><button type="button" class="button button1">Recordings</button></a>

</p>

<?php
$myVideoDir = '/var/www/camara1/streamer';
$iplink = 'http://192.168.1.128/streamer';
$extension = 'mp4';
$videoFile = false;
$pseudoDir = array(
'.',
'..',
'somefile.txt',
'video.mp4'
);

// uncomment the line below to use on your site
$pseudoDir = scandir($myVideoDir);
foreach($pseudoDir as $item) {
    if ( $item != '..' && $item != '.' && !is_dir($item) ) {
        $ext = preg_replace('#^.*\.([a-zA-Z0-9]+)$#', '$1', $item);
        if ( $ext == $extension )
            $videoFile = $item;
    }
}

if ( !!$videoFile ) {
    echo '
        <video id="dep" class="center" width="700" height="500" autoplay Loop mute>        
          <source src="'.$iplink.'/'.$videoFile.'" type="video/mp4"> 
        </video>
	
	<meta http-equiv="refresh" content="60">
    ';
	
	
}
?>




        

</body>
</html>

