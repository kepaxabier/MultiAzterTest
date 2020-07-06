<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<!--
Design by TEMPLATED
http://templated.co
Released for free under the Creative Commons Attribution License

Name       : PlainDisplay
Description: A two-column, fixed-width design with dark color scheme.
Version    : 1.0
Released   : 20140309

-->
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
    <title>MultiazterTest</title>
    <script src="http://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js" type="text/javascript"></script>
    <script type="text/javascript">
        function processing() {
            document.getElementById("img_processing").style.display = "block";
        }

        function checkextension() {
            var file = document.querySelector("#infile");
            if (document.getElementById("infile").files.length == 0) {
                alert("You have to upload at least one file!");
                return false;
            } else {
                var index = 0;
                while (index < document.getElementById("infile").files.length) {
                    if (/\.(txt|docx|doc|odt)$/i.test(file.files[index].name) === false) {
                        alert("Invalid format. Only .txt, .odt, .doc or .docx files allowed.");
                        return false;
                    }
                    index++;
                }
                if (document.getElementById('similarity').checked && document.getElementById("infile").files.length > 5) {
                    alert("Invalid number of files. If you want to compute the Semantic Similarity, you cannot analyze more than 5 files at the same time.");
                    return false;
                }
            }
            document.getElementById("mensajeResultados").innerHTML = "Analyzing... This can take a few minutes.\n\nPlease, wait.";
            document.getElementById("resultados").innerHTML = "";
            processing();
        }
    </script>

    <meta name="keywords" content=""/>
    <meta name="description" content=""/>
    <link href="http://fonts.googleapis.com/css?family=Varela" rel="stylesheet"/>
    <link href="default.css" rel="stylesheet" type="text/css" media="all"/>
    <link href="fonts.css" rel="stylesheet" type="text/css" media="all"/>

    <!--[if IE 6]>
    <link href="default_ie6.css" rel="stylesheet" type="text/css"/><![endif]-->

</head>
<body>
<div id="wrapper">
    <div id="header-wrapper">
        <div id="header" class="container">
            <div id="logo">
                <h1><a href="index.php">MULTIAZTERTEST</a></h1>
            </div>
            <div id="menu">
                <ul>
                    <li class="current_page_item"><a href="index.php" accesskey="1" title="">ANALYZE</a></li>
                    <li><a href="information.html" accesskey="2" title="">KNOW MORE</a></li>
                </ul>
            </div>
        </div>
    </div>
    <div id="banner">
        <div class="container">
            <div class="title">
                <h2>ANALYZE DOCUMENTS</h2>
                <span class="byline">Select the files you want to analyze</span></div>
        </div>
        <ul>
            <form id="upload" enctype="multipart/form-data" method="post" onsubmit="return checkextension(); ">
                <li><input id="infile" name="infile[]" type="file" value='Seleccionar' multiple></li>
                <li><input type="submit" id="submit" name="submit" value="ANALYZE" class="button"></li>
                <br>
                <select name="select" id="select">
	            <option selected disabled hidden>Choose here the language of the text</option>
                    <option id="basque" name="basque" value="basque">Basque</option>
                    <option id="english" name="english" value="english">English</option>
                    <option id="spanish" name="spanish" value="spanish">Spanish</option>
                </select>
		<br>
		<br>
                <!--<div align="left" style="margin-left:700px"><input type="checkbox" id="prediction" name="prediction" value="prediction"> Check if you want to predict the complexity level of the text</div>-->
                <li><input type="checkbox" id="only-ratios" name="only-ratios" value="only-ratios"> Check if you want only ratios</input></li>

                <li> The information of every group is displayed by default. </li>
                <li> In case you want to select some groups only, check them manually:</li>
                
                <li><input type="checkbox" id="descriptive" name="descriptive" value="descriptive"> Descriptive</input></li>
                <li><input type="checkbox" id="lexical-diversity" name="lexical-diversity" value="lexical-diversity"> Lexical Diversity</input></li>
                <li><input type="checkbox" id="readability" name="readability" value="readability"> Readability ability</input></li>
                <li><input type="checkbox" id="word-freq" name="word-freq" value="word-freq"> Word Frequency</input></li>
                <li><input type="checkbox" id="vocabulary" name="vocabulary" value="vocabulary"> Vocabulary knowledge</input></li>
                <li><input type="checkbox" id="word-info" name="word-info" value="word-info"> Word Information</input></li>
                <li><input type="checkbox" id="syntactic" name="syntactic" value="syntactic"> Syntactic Complexity</input></li>
                <li><input type="checkbox" id="semantic-info" name="semantic-info" value="semantic-info"> Semantic Information</input></li>
                <li><input type="checkbox" id="cohesion" name="cohesion" value="cohesion"> Referential Cohesion</input></li>
                <li><input type="checkbox" id="semantic-overlap" name="semantic-overlap" value="semantic-overlap"> Semantic Overlap</input></li>
                <li><input type="checkbox" id="connectives" name="connectives" value="connectives"> Discourse Connectives</input></li>
            </form>
        </ul>
    </div>
    <div id="extra" class="container">
        <div class="title">
            <span id="mensajeResultados" class="byline">The results will appear here.</span></div>
        <div id="resultados">
            <?php
            ini_set('display_errors', 1);
            ini_set('display_startup_errors', 1);
            error_reporting(E_ALL);
            function is_dir_empty($dir)
            {
                if (!is_readable($dir)) return NULL;
                $handle = opendir($dir);
                while (false !== ($entry = readdir($handle))) {
                    if ($entry != "." && $entry != "..") {
                        return FALSE;
                    }
                }
                return TRUE;
            }

            $workdir = "/var/www/html/aztertest";
            $destDir = "/var/www/html/aztertest/webfiles";
            $binPath = "/var/www/html/aztertest/webprocess.sh";
            $fileId = date('Y-m-d_His_');
            // Execute this code if the submit button is pressed.
            if (isset($_POST['submit'])) {
		if(!isset($_POST['select'])){
		    echo '<script type="text/javascript">',
			 ' alert("Please, select a language.");',
			 '</script>'
	            ;
		    return;
		}
                $name = md5(rand() * time());
                $uploadDir = "/var/www/html/aztertest/uploads/" . $name;
                if (!is_dir($uploadDir)) {
                    mkdir($uploadDir, 0777, true);
                }
                for ($i = 0; $i < count($_FILES['infile']['name']); $i++) {
                    echo " <a href='#" . $i . "'>Results of " . $_FILES['infile']['name'][$i] . "</a><br>";
                    $moved = move_uploaded_file($_FILES['infile']['tmp_name'][$i], $uploadDir . "/" . $_FILES['infile']['name'][$i]);
                    if ($moved) {
                        echo "<br>";
                    } else {
                        echo "The files could not be loaded. <br>";
                    }
                }
                if (is_dir_empty("./" . $uploadDir)) {
                    echo "An error has happened. Please, try again.";
                } else {
                    $zip_abs_path = $workdir . '/downloads/Multiaztertest_' . $name . '.zip';
                    $zip = '/downloads/Multiaztertest_' . $name . '.zip';
                    $language = $_POST['select'];
                    $id_selection = '';

                    if (isset($_POST['descriptive'])){
                        $id_selection = $id_selection . '-id 1';
                    }
                    if (isset($_POST['lexical-diversity'])){
                        $id_selection = $id_selection . ' -id 2';
                    }
                    if (isset($_POST['readability'])){
                        $id_selection = $id_selection . ' -id 3';
                    }
                    if (isset($_POST['word-freq'])){
                        $id_selection = $id_selection . ' -id 4';
                    }
                    if (isset($_POST['vocabulary'])){
                        $id_selection = $id_selection . ' -id 5';
                    }
                    if (isset($_POST['word-info'])){
                        $id_selection = $id_selection . ' -id 6';
                    }
                    if (isset($_POST['syntactic'])){
                        $id_selection = $id_selection . ' -id 7';
                    }
                    if (isset($_POST['semantic-info'])){
                        $id_selection = $id_selection . ' -id 8';
                    }
                    if (isset($_POST['cohesion'])){
                        $id_selection = $id_selection . ' -id 9';
                    }
                    if (isset($_POST['semantic-overlap'])){
                        $id_selection = $id_selection . ' -id 10';
                    }
                    if (isset($_POST['connectives'])){
                        $id_selection = $id_selection . ' -id 11';
                    }
                    $ratios = '';
                    # ¿Ratios selected?
                    if (isset($_POST['only-ratios'])) {
                        $ratios = $ratios . '-r ';
                    }

                    # $ratios e $id_selection podrian ir vacios
                    $cmd = $binPath . " " . "'$uploadDir/*'" . " " . $zip_abs_path . " " . $uploadDir . " " . "'$ratios $id_selection'" .  " " . $language;

                    exec($cmd . " 2>&1", $output, $return);
                    echo "<script>$('#mensajeResultados').html('<a href=" . $zip . ">Download results</a>');</script>";
                    $counter = 0;

		    $cmd_remove = "rm -f " . $uploadDir . "/results/full_results_aztertest.csv";
		    exec($cmd_remove . " 2>&1", $output_rm, $return_rm);
                    foreach (new DirectoryIterator($uploadDir . "/results") as $fileInfo) {
                        if ($fileInfo->isDot()) continue;
                        echo "<table>";
                        echo "<thead>
      			<tr>
      			<th colspan='3' id='" . $counter . "'>File: " . $_FILES['infile']['name'][$counter] . "</th>
      			</tr>
      			</thead>";
                        $f = fopen($uploadDir . '/results' . '/' . $fileInfo->getFilename(), "r");
                        $first = true;
                        while (($line = fgetcsv($f, 0, ":")) !== false) {
                            echo "<tr>";
                            foreach ($line as $cell) {
                                if (count($line) == 1) {
                                    echo "<td colspan='3' id='titulo'>" . htmlspecialchars($cell) . "</tr>";
                                } else {
                                    if ($first) {
                                        $first = false;
                                        echo "<td id='level'>" . htmlspecialchars($cell) . "</td>";
                                    } else {
                                        if ($cell == " Elementary") {
                                            echo "<td id='elementary'>" . htmlspecialchars($cell) . "</td>";
                                        } elseif ($cell == " Intermediate") {
                                            echo "<td id='intermediate'>" . htmlspecialchars($cell) . "</td>";
                                        } elseif ($cell == " Advanced") {
                                            echo "<td id='advanced'>" . htmlspecialchars($cell) . "</td>";
                                        } else {
                                            echo "<td>" . htmlspecialchars($cell) . "</td>";
                                        }
                                    }
                                }
                            }

                            echo "</tr>\n";
                        }
                        fclose($f);
                        echo "\n</table><ul class='actions'>
        			<li><a href='#' class='button'>Go to the top</a></li>
        		</ul>";
                        $counter++;
                    }
                }

                // //Visualizar el contenido del inputtext
                // //echo ('Input text:&nbsp;');
                // //echo $_POST['inputtext'];
                // //$fn es un nombre aletorio del pelo de /tmp/2018-12-14_162700_GVC0f3 en la carpeta /tmp
                // //tempnam — Crea un fichero con un nombre de fichero único Descripción string tempnam ( string $dir , string $prefix )
                // //$fn = tempnam (sys_get_temp_dir(), $fileId);
                // $fn = tempnam ($destDir, $fileId);
                // /*echo "<p>";
                // echo ($fn);
                // echo "</p>";*/
                // // da permisos rw_r__r__
                // chmod ($fn,0664);
                // // Si lo ha creado vacio
                // if ($fn)
                // {
                // //abrimos
                // $f = fopen ($fn, "w");
                // if ($f)
                // {
                // //escibimos el conenido de la caja en el
                // fwrite($f,$_POST['inputtext']);
                // fwrite($f,"\n");
                //    //cerramos el fichero
                // fclose($f);
                //    //obtenemos el nombre del fichero sin path
                //    $basefn = basename($fn);
                //    //dos2unix - Convertidor de archivos de texto de formato DOS/Mac a Unix y viceversa
                // exec("/usr/bin/dos2unix ".$fn);
                #The php script will look for a script "x.sh" in the current directory
                #the script will run as the user of the web-server - typically "www-data".
                #Bai ->  $sysout = shell_exec("cp /tmp/".$basefn." ".$fn.".out.csv" );
                #$sysout = exec($binPath." /tmp/".$basefn);
                #Este me crea vacio!!!!!
                #exec("/var/www/html/webprocess.sh /var/www/html/webfiles/proba2.txt", $output, $return);


                #No ->   $sysout = shell_exec($binPath." /tmp/".$basefn);
                // $emaPath = $fn.".out.csv";
                //    /*echo "<p>";
                // echo ($emaPath);
                //    echo "</p>";*/
                //    //El fichero existe pero es de tamaño 0!!!!!
                // if (file_exists($emaPath)) //&& filesize($emaPath) > 0)
                //    {
                //      #file_get_contents — Transmite un fichero completo a una cadena
                //      chmod ($fn,0644);
                // $progema = file_get_contents($emaPath);
                //      /*echo "<p>";
                // echo ("Irteera fitxategiaren edukina:".$progema);
                //      echo "</p>";*/
                // $progema_web="";
                //      # array explode ( string $delimiter , string $string [, int $limit = PHP_INT_MAX ] )
                // # Devuelve un array de string, siendo cada uno un substring del parámetro string formado por la división realizada por los delimitadores indicados en el parámetro delimiter.
                // $progema_array = explode("\n", $progema);
                // for ($i=0; $i < (count($progema_array)); $i++) {
                //        #Por cada linea: delimitador de columna ","
                //        #$line_array=explode(",", $progema_array[$i]);
                // # mixed preg_replace ( mixed $pattern , mixed $replacement , mixed $subject [, int $limit = -1 [, int &$count ]] )
                // #Busca en subject coincidencias de pattern y las reemplaza con replacement.
                //        #$lerroa = preg_replace('/^[0-9]+\,/', '', $progema_array[$i]);
                //        #Si es la primera línea
                //        #if ($line_array[0] ==1)
                // #$progema_web=$progema_web."<b>".$progema_array[i]."</b><br>";}
                // #else
                //        #parrafoka
                // $progema_web=$progema_web.$progema_array[$i]."<br>";
                //
                // }
                // echo "<div id='emadiv' style='text-align: justify;'>".$progema_web."</div>";
                // }
                // else
                // {
                // echo "<p>Erroreren bat egon da testua analizatzean.</p>";
                // }
                //
                //
                // }
                //
                // }
                // else
                // { echo("<p>Erroreren bat egon da fitxategi tenporala sortzean.</p>");}
                //

            }

            ?>
        </div>
        <br>

        <div id="img_processing" style="display:none;margin:auto;"><img src="processing.gif" alt=""/><br>Processing...
        </div>
    </div>
</div>
</div>


<div id="copyright" class="container">
    <p>&copy; MultiAzterTest. All rights reserved. | Template by <a href="http://templated.co" rel="nofollow">TEMPLATED</a>.
    </p>
</div>
</body>
</html>
