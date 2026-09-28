const fileInput = document.querySelector('#file-upload');
const fileName = document.querySelector('#file-name');
const fileButton = document.querySelector('#file-button');

fileInput.addEventListener('change', () => {

    const file = fileInput.files[0];

    if (file) {
        fileName.textContent = file.name;
        fileButton.textContent = '選択済み';
    } else {
        fileName.textContent = 'ファイルが選択されていません';
        fileButton.textContent = 'ファイルを選択';
    }

});