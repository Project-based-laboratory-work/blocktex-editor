const fileInput = document.querySelector<HTMLInputElement>('#file-upload');
const fileName = document.querySelector<HTMLElement>('#file-name');
const fileButton = document.querySelector<HTMLElement>('#file-button');

if (!fileInput || !fileName || !fileButton) {
    throw new Error('必要な要素が見つかりません: #file-upload, #file-name, #file-button');
}

fileInput.addEventListener('change', () => {

    const file = fileInput.files?.[0];

    if (file) {
        fileName.textContent = file.name;
        fileButton.textContent = '選択済み';
    } else {
        fileName.textContent = 'ファイルが選択されていません';
        fileButton.textContent = 'ファイルを選択';
    }

});
