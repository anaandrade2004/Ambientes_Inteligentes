function delete_conceito(designacao) {
    $.ajax("/conceitos/"+ designacao, {
        type: "DELETE",
        success: function(data){
            console.log(data);
            if (data["success"]){
                window.location.href=data["redirect_url"]
            }
        },
        error: function(data){
            console.log(error);
        }
    });
}

$(document).ready( function () {
    $('#tabela_conceitos').DataTable();
} );