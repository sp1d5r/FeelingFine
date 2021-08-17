
window.onload=function(){
    const actualBtn = document.getElementById('actual-btn');
    console.log(actualBtn)

    const fileChosen = document.getElementById('file-chosen');
    console.log(fileChosen)

    actualBtn.addEventListener('change', function(){
        console.log(fileChosen.textContent);
        fileChosen.textContent = this.files[0].name
    })
}