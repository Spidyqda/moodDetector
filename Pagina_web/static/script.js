document.addEventListener('DOMContentLoaded', (event) => {
    const enableButton = document.getElementById('enableWrite');
    const disableButton = document.getElementById('disableWrite');

    enableButton.addEventListener('click', () => {
        $.post("/enable_write_to_file", function (data) {
            alert(data.message);
        });
    });

    disableButton.addEventListener('click', () => {
        $.post("/disable_write_to_file", function (data) {
            alert(data.message);
        });
    });

});
