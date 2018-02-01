import { h, app } from 'hyperapp'
import { Link, Router, Route, location } from '@hyperapp/router'

type State = typeof state;
type Actions = typeof actions;

const state = {
  vega: {},
  conut: 0,
  isFetching: false,
  filename: '',
}

const actions = {
  inc: () => (state: State, actions:Actions) => {
    return {
      count: state.count + 1
    }
  },
  toggleFetching: isFetching => {
    isFetching: isFetching
  },
  setFile: filename => ({filename:filename}),
  setVega: vega_data => ({vega:vega_data}),
  setData: () => async (state: State, actions:Actions) => {
    actions.toggleFetching(true)
    const url = `http://127.0.0.1:4567/last-vega`
    const response = await fetch(url)
    const data = await response.json()
    actions.toggleFetching(false)
    actions.setFile(data.filename)
    actions.setVega(data.vega_data)

    console.log('data', data)
    console.log('http://localhost:4567/uploads/'+state.filename)
    console.log(state, ',,,,,,,,,')
    vegaEmbed("#vega", data.vega);
  },
}

const Graph = (state:State, actions:Actions) =>
  <div onupdate={actions.setData}>
    <Link to={`http://localhost:4567`}><h1>Cami & co.</h1></Link>
    <img width=300 src={'http://localhost:4567/uploads/'+state.filename} />
    <div id='vega'/>
  </div>

const view = (state:State, actions:Actions) => 
  <div onupdate={actions.setData}>
    <Route path='/' render={()=> <Link to={'http://localhost:4567'}>btn</Link>} />
    <Route path='/vega' render={()=> <Graph state={state} actions={actions} /> }/>
  </div>

const main = app(state, actions, view, document.body)
