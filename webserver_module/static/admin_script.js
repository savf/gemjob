var refreshInterval = 10000;
var intervalId = -1;

$(document).ready(function() {

    $('#GetSample').bind('click', function(e) {
        $("#Result").text("Requesting new sample ...");
        $('#GetSample').prop("disabled",true).addClass("Disabled");

        $.getJSON($SCRIPT_ROOT + '/get_sample', {
            sample_size: $('input[name="sample_size"]').val(),
            days_posted: $('input[name="days_posted"]').val(),
            page_offset: $('input[name="page_offset"]').val()
        }).done(function(data) {
            $('#GetSample').prop("disabled",false).removeClass("Disabled");
            $("#Result").text(data.result);
        }).fail(function( jqxhr, textStatus, error ) {
            $('#GetSample').prop("disabled",false).removeClass("Disabled");
            $("#Result").text("Failed");
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
    $.getJSON($SCRIPT_ROOT + '/is_online', {}).done( function (data) {
        $("#WS").text(data.result['WS']);
        $("#WS").addClass("OK").removeClass("Warning");

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
    }).fail(function( jqxhr, textStatus, error ) {
            $("#WS").text('offline');
            $("#WS").addClass("Warning").removeClass("OK");
            $("#D").text('offline');
            $("#D").addClass("Warning").removeClass("OK");
            $("#DM").text('offline');
            $("#DM").addClass("Warning").removeClass("OK");
            $("#DB").text('offline');
            $("#DB").addClass("Warning").removeClass("OK");
        });
}