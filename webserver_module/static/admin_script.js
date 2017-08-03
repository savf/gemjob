var refreshInterval = 10000;
var intervalId = -1;
var module_ids = ['WS', 'D', 'DB', 'CL', 'BU', 'FE', 'JO', 'KNN'];

$(document).ready(function() {

    check_module_status();
    intervalId = window.setInterval(function(){
        check_module_status();
        }, refreshInterval);

});

function check_module_status() {
    $.getJSON($SCRIPT_ROOT + '/is_online', {}).done( function (data) {

        for (var i = 0; i < module_ids.length; i++) {
            $("#"+module_ids[i]).text(data.result[module_ids[i]]);
            if (data.result[module_ids[i]] != 'online')
                $("#"+module_ids[i]).addClass("Warning").removeClass("OK");
            else
                $("#"+module_ids[i]).removeClass("Warning").addClass("OK");
        }

    }).fail(function( jqxhr, textStatus, error ) {
        for (var i = 0; i < module_ids.length; i++) {
            $("#"+module_ids[i]).text('offline');
            $("#"+module_ids[i]).addClass("Warning").removeClass("OK");
        }
    });
}
