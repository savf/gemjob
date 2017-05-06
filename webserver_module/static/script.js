var refreshInterval = 10000;
var intervalId = -1;

$(document).ready(function() {

    $('#GetSample').bind('click', function(e) {
        $("#Result").text("Requesting new sample ...");
        $('#GetSample').prop("disabled",true).addClass("Disabled");

        $.getJSON($SCRIPT_ROOT + '/get_sample', {
            sample_size: $('input[name="sample_size"]').val()
        }, function(data) {
            $('#GetSample').prop("disabled",false).removeClass("Disabled");
            $("#Result").text(data.result);
        });
        e.preventDefault();
        return false;
    });

    check_module_status();
    intervalId = window.setInterval(function(){
        check_module_status();
        }, refreshInterval);

});

function check_module_status() {
    $.getJSON($SCRIPT_ROOT + '/is_online', {}, function (data) {
        $("#D").text(data.result['D']);
        if (data.result['D'] != 'online')
            $("#D").addClass("Warning").removeClass("OK");
        else
            $("#D").removeClass("Warning").addClass("OK");

        $("#DM").text(data.result['DM']);
        if (data.result['DM'] != 'online')
            $("#DM").addClass("Warning").removeClass("OK");
        else
            $("#DM").removeClass("Warning").addClass("OK");

        $("#DB").text(data.result['DB']);
        if (data.result['DB'] != 'online')
            $("#DB").addClass("Warning").removeClass("OK");
        else
            $("#DB").removeClass("Warning").addClass("OK");
    });
}