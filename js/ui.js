function gradioApp() {
    const elems = document.getElementsByTagName('gradio-app');
    const elem = elems.length == 0 ? document : elems[0];

    if (elem !== document) {
        elem.getElementById = function(id) {
            return document.getElementById(id);
        };
    }
    return elem.shadowRoot ? elem.shadowRoot : elem;
}

/**
 * Get the currently selected top-level UI tab button (e.g. the button that says "Extras").
 */
function get_uiCurrentTab() {
    return gradioApp().querySelector('#tabs > .tab-nav > button.selected');
}


const uiUpdateCallbacks = [];
const uiAfterUpdateCallbacks = [];
const uiLoadedCallbacks = [];
const uiTabChangeCallbacks = [];
let uiAfterUpdateTimeout = null;
let uiCurrentTab = null;

function onUiLoaded(callback) {
    uiLoadedCallbacks.push(callback);
}


function executeCallbacks(queue, arg) {
    for (const callback of queue) {
        try {
            callback(arg);
        } catch (e) {
            console.error("error running callback", callback, ":", e);
        }
    }
}

/**
 * Schedule the execution of the callbacks registered with onAfterUiUpdate.
 * The callbacks are executed after a short while, unless another call to this function
 * is made before that time. IOW, the callbacks are executed only once, even
 * when there are multiple mutations observed.
 */
function scheduleAfterUiUpdateCallbacks() {
    clearTimeout(uiAfterUpdateTimeout);
    uiAfterUpdateTimeout = setTimeout(function() {
        executeCallbacks(uiAfterUpdateCallbacks);
    }, 200);
}

let executedOnLoaded = false;


document.addEventListener("DOMContentLoaded", function() {
    const mutationObserver = new MutationObserver(function (m) {
        if (!executedOnLoaded && gradioApp().querySelector('#processor_list')) {
            executedOnLoaded = true;
            executeCallbacks(uiLoadedCallbacks);
        }

        executeCallbacks(uiUpdateCallbacks, m);
        scheduleAfterUiUpdateCallbacks();
        const newTab = get_uiCurrentTab();
        if (newTab && (newTab !== uiCurrentTab)) {
            uiCurrentTab = newTab;
            executeCallbacks(uiTabChangeCallbacks);
        }
    });

    mutationObserver.observe(gradioApp(), {childList: true, subtree: true});
});

function refresh() {
    const url = new URL(window.location);
    console.log("REFRESH:", url);
    // if (url.searchParams.get('__theme') !== 'dark') {
    //     url.searchParams.set('__theme', 'dark');
    //     window.location.href = url.href;
    // }
}
