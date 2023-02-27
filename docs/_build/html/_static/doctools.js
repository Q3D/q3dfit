/*
 * doctools.js
 * ~~~~~~~~~~~
 *
<<<<<<< HEAD
 * Base JavaScript utilities for all Sphinx HTML documentation.
=======
 * Sphinx JavaScript utilities for all documentation.
>>>>>>> public/v1.1.0
 *
 * :copyright: Copyright 2007-2022 by the Sphinx team, see AUTHORS.
 * :license: BSD, see LICENSE for details.
 *
 */
<<<<<<< HEAD
"use strict";

const _ready = (callback) => {
  if (document.readyState !== "loading") {
    callback();
  } else {
    document.addEventListener("DOMContentLoaded", callback);
  }
};

/**
 * highlight a given string on a node by wrapping it in
 * span elements with the given class name.
 */
const _highlight = (node, addItems, text, className) => {
  if (node.nodeType === Node.TEXT_NODE) {
    const val = node.nodeValue;
    const parent = node.parentNode;
    const pos = val.toLowerCase().indexOf(text);
    if (
      pos >= 0 &&
      !parent.classList.contains(className) &&
      !parent.classList.contains("nohighlight")
    ) {
      let span;

      const closestNode = parent.closest("body, svg, foreignObject");
      const isInSVG = closestNode && closestNode.matches("svg");
      if (isInSVG) {
        span = document.createElementNS("http://www.w3.org/2000/svg", "tspan");
      } else {
        span = document.createElement("span");
        span.classList.add(className);
      }

      span.appendChild(document.createTextNode(val.substr(pos, text.length)));
      parent.insertBefore(
        span,
        parent.insertBefore(
          document.createTextNode(val.substr(pos + text.length)),
          node.nextSibling
        )
      );
      node.nodeValue = val.substr(0, pos);

      if (isInSVG) {
        const rect = document.createElementNS(
          "http://www.w3.org/2000/svg",
          "rect"
        );
        const bbox = parent.getBBox();
        rect.x.baseVal.value = bbox.x;
        rect.y.baseVal.value = bbox.y;
        rect.width.baseVal.value = bbox.width;
        rect.height.baseVal.value = bbox.height;
        rect.setAttribute("class", className);
        addItems.push({ parent: parent, target: rect });
      }
    }
  } else if (node.matches && !node.matches("button, select, textarea")) {
    node.childNodes.forEach((el) => _highlight(el, addItems, text, className));
  }
};
const _highlightText = (thisNode, text, className) => {
  let addItems = [];
  _highlight(thisNode, addItems, text, className);
  addItems.forEach((obj) =>
    obj.parent.insertAdjacentElement("beforebegin", obj.target)
  );
};
=======

/**
 * select a different prefix for underscore
 */
$u = _.noConflict();

/**
 * make the code below compatible with browsers without
 * an installed firebug like debugger
if (!window.console || !console.firebug) {
  var names = ["log", "debug", "info", "warn", "error", "assert", "dir",
    "dirxml", "group", "groupEnd", "time", "timeEnd", "count", "trace",
    "profile", "profileEnd"];
  window.console = {};
  for (var i = 0; i < names.length; ++i)
    window.console[names[i]] = function() {};
}
 */

/**
 * small helper function to urldecode strings
 *
 * See https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/decodeURIComponent#Decoding_query_parameters_from_a_URL
 */
jQuery.urldecode = function(x) {
  if (!x) {
    return x
  }
  return decodeURIComponent(x.replace(/\+/g, ' '));
};

/**
 * small helper function to urlencode strings
 */
jQuery.urlencode = encodeURIComponent;

/**
 * This function returns the parsed url parameters of the
 * current request. Multiple values per key are supported,
 * it will always return arrays of strings for the value parts.
 */
jQuery.getQueryParameters = function(s) {
  if (typeof s === 'undefined')
    s = document.location.search;
  var parts = s.substr(s.indexOf('?') + 1).split('&');
  var result = {};
  for (var i = 0; i < parts.length; i++) {
    var tmp = parts[i].split('=', 2);
    var key = jQuery.urldecode(tmp[0]);
    var value = jQuery.urldecode(tmp[1]);
    if (key in result)
      result[key].push(value);
    else
      result[key] = [value];
  }
  return result;
};

/**
 * highlight a given string on a jquery object by wrapping it in
 * span elements with the given class name.
 */
jQuery.fn.highlightText = function(text, className) {
  function highlight(node, addItems) {
    if (node.nodeType === 3) {
      var val = node.nodeValue;
      var pos = val.toLowerCase().indexOf(text);
      if (pos >= 0 &&
          !jQuery(node.parentNode).hasClass(className) &&
          !jQuery(node.parentNode).hasClass("nohighlight")) {
        var span;
        var isInSVG = jQuery(node).closest("body, svg, foreignObject").is("svg");
        if (isInSVG) {
          span = document.createElementNS("http://www.w3.org/2000/svg", "tspan");
        } else {
          span = document.createElement("span");
          span.className = className;
        }
        span.appendChild(document.createTextNode(val.substr(pos, text.length)));
        node.parentNode.insertBefore(span, node.parentNode.insertBefore(
          document.createTextNode(val.substr(pos + text.length)),
          node.nextSibling));
        node.nodeValue = val.substr(0, pos);
        if (isInSVG) {
          var rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
          var bbox = node.parentElement.getBBox();
          rect.x.baseVal.value = bbox.x;
          rect.y.baseVal.value = bbox.y;
          rect.width.baseVal.value = bbox.width;
          rect.height.baseVal.value = bbox.height;
          rect.setAttribute('class', className);
          addItems.push({
              "parent": node.parentNode,
              "target": rect});
        }
      }
    }
    else if (!jQuery(node).is("button, select, textarea")) {
      jQuery.each(node.childNodes, function() {
        highlight(this, addItems);
      });
    }
  }
  var addItems = [];
  var result = this.each(function() {
    highlight(this, addItems);
  });
  for (var i = 0; i < addItems.length; ++i) {
    jQuery(addItems[i].parent).before(addItems[i].target);
  }
  return result;
};

/*
 * backward compatibility for jQuery.browser
 * This will be supported until firefox bug is fixed.
 */
if (!jQuery.browser) {
  jQuery.uaMatch = function(ua) {
    ua = ua.toLowerCase();

    var match = /(chrome)[ \/]([\w.]+)/.exec(ua) ||
      /(webkit)[ \/]([\w.]+)/.exec(ua) ||
      /(opera)(?:.*version|)[ \/]([\w.]+)/.exec(ua) ||
      /(msie) ([\w.]+)/.exec(ua) ||
      ua.indexOf("compatible") < 0 && /(mozilla)(?:.*? rv:([\w.]+)|)/.exec(ua) ||
      [];

    return {
      browser: match[ 1 ] || "",
      version: match[ 2 ] || "0"
    };
  };
  jQuery.browser = {};
  jQuery.browser[jQuery.uaMatch(navigator.userAgent).browser] = true;
}
>>>>>>> public/v1.1.0

/**
 * Small JavaScript module for the documentation.
 */
<<<<<<< HEAD
const Documentation = {
  init: () => {
    Documentation.highlightSearchWords();
    Documentation.initDomainIndexTable();
    Documentation.initOnKeyListeners();
=======
var Documentation = {

  init : function() {
    this.fixFirefoxAnchorBug();
    this.highlightSearchWords();
    this.initIndexTable();
    this.initOnKeyListeners();
>>>>>>> public/v1.1.0
  },

  /**
   * i18n support
   */
<<<<<<< HEAD
  TRANSLATIONS: {},
  PLURAL_EXPR: (n) => (n === 1 ? 0 : 1),
  LOCALE: "unknown",

  // gettext and ngettext don't access this so that the functions
  // can safely bound to a different name (_ = Documentation.gettext)
  gettext: (string) => {
    const translated = Documentation.TRANSLATIONS[string];
    switch (typeof translated) {
      case "undefined":
        return string; // no translation
      case "string":
        return translated; // translation exists
      default:
        return translated[0]; // (singular, plural) translation tuple exists
    }
  },

  ngettext: (singular, plural, n) => {
    const translated = Documentation.TRANSLATIONS[singular];
    if (typeof translated !== "undefined")
      return translated[Documentation.PLURAL_EXPR(n)];
    return n === 1 ? singular : plural;
  },

  addTranslations: (catalog) => {
    Object.assign(Documentation.TRANSLATIONS, catalog.messages);
    Documentation.PLURAL_EXPR = new Function(
      "n",
      `return (${catalog.plural_expr})`
    );
    Documentation.LOCALE = catalog.locale;
=======
  TRANSLATIONS : {},
  PLURAL_EXPR : function(n) { return n === 1 ? 0 : 1; },
  LOCALE : 'unknown',

  // gettext and ngettext don't access this so that the functions
  // can safely bound to a different name (_ = Documentation.gettext)
  gettext : function(string) {
    var translated = Documentation.TRANSLATIONS[string];
    if (typeof translated === 'undefined')
      return string;
    return (typeof translated === 'string') ? translated : translated[0];
  },

  ngettext : function(singular, plural, n) {
    var translated = Documentation.TRANSLATIONS[singular];
    if (typeof translated === 'undefined')
      return (n == 1) ? singular : plural;
    return translated[Documentation.PLURALEXPR(n)];
  },

  addTranslations : function(catalog) {
    for (var key in catalog.messages)
      this.TRANSLATIONS[key] = catalog.messages[key];
    this.PLURAL_EXPR = new Function('n', 'return +(' + catalog.plural_expr + ')');
    this.LOCALE = catalog.locale;
  },

  /**
   * add context elements like header anchor links
   */
  addContextElements : function() {
    $('div[id] > :header:first').each(function() {
      $('<a class="headerlink">\u00B6</a>').
      attr('href', '#' + this.id).
      attr('title', _('Permalink to this headline')).
      appendTo(this);
    });
    $('dt[id]').each(function() {
      $('<a class="headerlink">\u00B6</a>').
      attr('href', '#' + this.id).
      attr('title', _('Permalink to this definition')).
      appendTo(this);
    });
  },

  /**
   * workaround a firefox stupidity
   * see: https://bugzilla.mozilla.org/show_bug.cgi?id=645075
   */
  fixFirefoxAnchorBug : function() {
    if (document.location.hash && $.browser.mozilla)
      window.setTimeout(function() {
        document.location.href += '';
      }, 10);
>>>>>>> public/v1.1.0
  },

  /**
   * highlight the search words provided in the url in the text
   */
<<<<<<< HEAD
  highlightSearchWords: () => {
    const highlight =
      new URLSearchParams(window.location.search).get("highlight") || "";
    const terms = highlight.toLowerCase().split(/\s+/).filter(x => x);
    if (terms.length === 0) return; // nothing to do

    // There should never be more than one element matching "div.body"
    const divBody = document.querySelectorAll("div.body");
    const body = divBody.length ? divBody[0] : document.querySelector("body");
    window.setTimeout(() => {
      terms.forEach((term) => _highlightText(body, term, "highlighted"));
    }, 10);

    const searchBox = document.getElementById("searchbox");
    if (searchBox === null) return;
    searchBox.appendChild(
      document
        .createRange()
        .createContextualFragment(
          '<p class="highlight-link">' +
            '<a href="javascript:Documentation.hideSearchWords()">' +
            Documentation.gettext("Hide Search Matches") +
            "</a></p>"
        )
    );
=======
  highlightSearchWords : function() {
    var params = $.getQueryParameters();
    var terms = (params.highlight) ? params.highlight[0].split(/\s+/) : [];
    if (terms.length) {
      var body = $('div.body');
      if (!body.length) {
        body = $('body');
      }
      window.setTimeout(function() {
        $.each(terms, function() {
          body.highlightText(this.toLowerCase(), 'highlighted');
        });
      }, 10);
      $('<p class="highlight-link"><a href="javascript:Documentation.' +
        'hideSearchWords()">' + _('Hide Search Matches') + '</a></p>')
          .appendTo($('#searchbox'));
    }
  },

  /**
   * init the domain index toggle buttons
   */
  initIndexTable : function() {
    var togglers = $('img.toggler').click(function() {
      var src = $(this).attr('src');
      var idnum = $(this).attr('id').substr(7);
      $('tr.cg-' + idnum).toggle();
      if (src.substr(-9) === 'minus.png')
        $(this).attr('src', src.substr(0, src.length-9) + 'plus.png');
      else
        $(this).attr('src', src.substr(0, src.length-8) + 'minus.png');
    }).css('display', '');
    if (DOCUMENTATION_OPTIONS.COLLAPSE_INDEX) {
        togglers.click();
    }
>>>>>>> public/v1.1.0
  },

  /**
   * helper function to hide the search marks again
   */
<<<<<<< HEAD
  hideSearchWords: () => {
    document
      .querySelectorAll("#searchbox .highlight-link")
      .forEach((el) => el.remove());
    document
      .querySelectorAll("span.highlighted")
      .forEach((el) => el.classList.remove("highlighted"));
    const url = new URL(window.location);
    url.searchParams.delete("highlight");
    window.history.replaceState({}, "", url);
  },

  /**
   * helper function to focus on search bar
   */
  focusSearchBar: () => {
    document.querySelectorAll("input[name=q]")[0]?.focus();
  },

  /**
   * Initialise the domain index toggle buttons
   */
  initDomainIndexTable: () => {
    const toggler = (el) => {
      const idNumber = el.id.substr(7);
      const toggledRows = document.querySelectorAll(`tr.cg-${idNumber}`);
      if (el.src.substr(-9) === "minus.png") {
        el.src = `${el.src.substr(0, el.src.length - 9)}plus.png`;
        toggledRows.forEach((el) => (el.style.display = "none"));
      } else {
        el.src = `${el.src.substr(0, el.src.length - 8)}minus.png`;
        toggledRows.forEach((el) => (el.style.display = ""));
      }
    };

    const togglerElements = document.querySelectorAll("img.toggler");
    togglerElements.forEach((el) =>
      el.addEventListener("click", (event) => toggler(event.currentTarget))
    );
    togglerElements.forEach((el) => (el.style.display = ""));
    if (DOCUMENTATION_OPTIONS.COLLAPSE_INDEX) togglerElements.forEach(toggler);
  },

  initOnKeyListeners: () => {
    // only install a listener if it is really needed
    if (
      !DOCUMENTATION_OPTIONS.NAVIGATION_WITH_KEYS &&
      !DOCUMENTATION_OPTIONS.ENABLE_SEARCH_SHORTCUTS
    )
      return;

    const blacklistedElements = new Set([
      "TEXTAREA",
      "INPUT",
      "SELECT",
      "BUTTON",
    ]);
    document.addEventListener("keydown", (event) => {
      if (blacklistedElements.has(document.activeElement.tagName)) return; // bail for input elements
      if (event.altKey || event.ctrlKey || event.metaKey) return; // bail with special keys

      if (!event.shiftKey) {
        switch (event.key) {
          case "ArrowLeft":
            if (!DOCUMENTATION_OPTIONS.NAVIGATION_WITH_KEYS) break;

            const prevLink = document.querySelector('link[rel="prev"]');
            if (prevLink && prevLink.href) {
              window.location.href = prevLink.href;
              event.preventDefault();
            }
            break;
          case "ArrowRight":
            if (!DOCUMENTATION_OPTIONS.NAVIGATION_WITH_KEYS) break;

            const nextLink = document.querySelector('link[rel="next"]');
            if (nextLink && nextLink.href) {
              window.location.href = nextLink.href;
              event.preventDefault();
            }
            break;
          case "Escape":
            if (!DOCUMENTATION_OPTIONS.ENABLE_SEARCH_SHORTCUTS) break;
            Documentation.hideSearchWords();
            event.preventDefault();
        }
      }

      // some keyboard layouts may need Shift to get /
      switch (event.key) {
        case "/":
          if (!DOCUMENTATION_OPTIONS.ENABLE_SEARCH_SHORTCUTS) break;
          Documentation.focusSearchBar();
          event.preventDefault();
      }
    });
  },
};

// quick alias for translations
const _ = Documentation.gettext;

_ready(Documentation.init);
=======
  hideSearchWords : function() {
    $('#searchbox .highlight-link').fadeOut(300);
    $('span.highlighted').removeClass('highlighted');
    var url = new URL(window.location);
    url.searchParams.delete('highlight');
    window.history.replaceState({}, '', url);
  },

   /**
   * helper function to focus on search bar
   */
  focusSearchBar : function() {
    $('input[name=q]').first().focus();
  },

  /**
   * make the url absolute
   */
  makeURL : function(relativeURL) {
    return DOCUMENTATION_OPTIONS.URL_ROOT + '/' + relativeURL;
  },

  /**
   * get the current relative url
   */
  getCurrentURL : function() {
    var path = document.location.pathname;
    var parts = path.split(/\//);
    $.each(DOCUMENTATION_OPTIONS.URL_ROOT.split(/\//), function() {
      if (this === '..')
        parts.pop();
    });
    var url = parts.join('/');
    return path.substring(url.lastIndexOf('/') + 1, path.length - 1);
  },

  initOnKeyListeners: function() {
    // only install a listener if it is really needed
    if (!DOCUMENTATION_OPTIONS.NAVIGATION_WITH_KEYS &&
        !DOCUMENTATION_OPTIONS.ENABLE_SEARCH_SHORTCUTS)
        return;

    $(document).keydown(function(event) {
      var activeElementType = document.activeElement.tagName;
      // don't navigate when in search box, textarea, dropdown or button
      if (activeElementType !== 'TEXTAREA' && activeElementType !== 'INPUT' && activeElementType !== 'SELECT'
          && activeElementType !== 'BUTTON') {
        if (event.altKey || event.ctrlKey || event.metaKey)
          return;

          if (!event.shiftKey) {
            switch (event.key) {
              case 'ArrowLeft':
                if (!DOCUMENTATION_OPTIONS.NAVIGATION_WITH_KEYS)
                  break;
                var prevHref = $('link[rel="prev"]').prop('href');
                if (prevHref) {
                  window.location.href = prevHref;
                  return false;
                }
                break;
              case 'ArrowRight':
                if (!DOCUMENTATION_OPTIONS.NAVIGATION_WITH_KEYS)
                  break;
                var nextHref = $('link[rel="next"]').prop('href');
                if (nextHref) {
                  window.location.href = nextHref;
                  return false;
                }
                break;
              case 'Escape':
                if (!DOCUMENTATION_OPTIONS.ENABLE_SEARCH_SHORTCUTS)
                  break;
                Documentation.hideSearchWords();
                return false;
          }
        }

        // some keyboard layouts may need Shift to get /
        switch (event.key) {
          case '/':
            if (!DOCUMENTATION_OPTIONS.ENABLE_SEARCH_SHORTCUTS)
              break;
            Documentation.focusSearchBar();
            return false;
        }
      }
    });
  }
};

// quick alias for translations
_ = Documentation.gettext;

$(document).ready(function() {
  Documentation.init();
});
>>>>>>> public/v1.1.0
