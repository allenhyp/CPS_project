<?php
$firstName = $_POST["firstName"];
$lastName = $_POST["lastName"];
$userId = $_POST["userId"];

$target_dir = "/home/allenhsu/uploads";if(!file_exists($target_dir))
{
echo json_encode([
"Message" => "Creating new dir.",
"Status" => "OK",
"userId" => $_REQUEST["userId"]
]);
mkdir($target_dir, 0777, true);
}

$target_dir = $target_dir . "/" . basename($_FILES["file"]["name"]);

if (move_uploaded_file($_FILES["file"]["tmp_name"], $target_dir)) 
{
putenv("PATH=/usr/local/bin/:" . exec('echo $PATH'));
$executePython = "/usr/bin/python3 tsc_cnn.py 2>&1";
$output = shell_exec("$executePython");
// exec('python3 tsc_cnn.py 2>&1', $output);

echo json_encode([
"Message" => "The file ". basename( $_FILES["file"]["name"]). " has been uploaded.",
"Result" => $output,
"Status" => "OK",
"userId" => $_REQUEST["userId"]
]);
pclose($handle);

} else {

echo json_encode([
"Message" => "Sorry, there was an error uploading your file.",
"Status" => "Error",
"userId" => $_REQUEST["userId"]
]);

}
?>

