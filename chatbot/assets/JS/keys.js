var timer = null;
$('input').keydown(function(){
        typehere = document.getElementById("typinghere");
        typehere.innerHTML = "<h5> You are typing ...</h5>";
         clearTimeout(timer);
         timer = setTimeout(doStuff, 1000)
});

  function doStuff() {
    typehere = document.getElementById("typinghere");
    typehere.innerHTML = "";
  }
