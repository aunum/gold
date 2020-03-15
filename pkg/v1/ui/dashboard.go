// Package ui package contains resources for creating agent dashboards.
package ui

import (
	"net/http"

	rice "github.com/GeertJohan/go.rice"
)

// ApplyHandlers adds the base handlers.
func ApplyHandlers(mux *http.ServeMux) error {
	box := rice.MustFindBox("./static")
	str, err := box.String("dashboard.html")
	if err != nil {
		return err
	}
	d := &dashboard{index: str}
	mux.HandleFunc("/", d.handler)
	return nil
}

type dashboard struct {
	index string
}

func (d *dashboard) handler(w http.ResponseWriter, req *http.Request) {
	w.WriteHeader(200)
	w.Write([]byte(d.index))
}
