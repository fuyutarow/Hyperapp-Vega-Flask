import { h, app } from 'hyperapp'
import { Link, Router, Route, location } from '@hyperapp/router'

type State = typeof state;
type Actions = typeof actions;

const state = {
  vega: {},
  conut: 0,
  isFetching: false,
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
  setData: () => async (state: State, actions:Actions) => {
    actions.toggleFetching(true)
    const url = `http://127.0.0.1:4567/last-vega`
    const response = await fetch(url)
    const data = await response.json()
    actions.toggleFetching(false)

    console.log('data', data)
    vegaEmbed("#vega", data.vega);
    return {
      vega: data.vega, filename: data.path
    }
  },
}

const Graph = (state:State, actions:Actions) =>
  <div>
    <Link to={`http://localhost:4567`}><h1>Cami & co.</h1></Link>
    <div id='vega'/>
  </div>

const view = (state:State, actions:Actions) => 
  <div onupdate={actions.setData}>
    <Route path='/' render={()=> <Link to={`http://localhost:4567`}>btn</Link>} />
    <Route path='/vega' render={()=> <Graph state={state} actions={actions} /> }/>
  </div>

const main = app(state, actions, view, document.body)
