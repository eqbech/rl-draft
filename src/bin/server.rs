use actix_cors::Cors;
use actix_web::{get, web, App, HttpResponse, HttpServer, Responder};
use rust_rl::{agents::q_agent::QAgent, Agent, GRID_SIZE};
use std::sync::Mutex;

// async fn guess_handler(query: web::Query<Request>, data: web::Data<Mutex<Agent>>) -> impl Responder {
//     // Lock the agent and generate a guess for the provided state.
//     let agent = data.lock().unwrap();
//     let guess = agent.predict(&query.state);

//     HttpResponse::Ok().json(guess)
// }
#[get("/q_table")]
async fn get_q_table(data: web::Data<Mutex<QAgent>>) -> impl Responder {
    // Lock the agent and retrieve the Q-table.
    let agent = data.lock().unwrap();
    HttpResponse::Ok().json(agent.q_table_to_2_dim_grid())
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Train the agent or load a saved Q-table.
    let agent = QAgent::load("data/q_table.csv", GRID_SIZE.0, GRID_SIZE.1)
        .expect("Failed to load Agent with Q-table");
    println!("Agent loaded with Q-table.");

    // Wrap the agent in a Mutex and web::Data to share state safely.
    let agent_data = web::Data::new(Mutex::new(agent));

    println!("Starting server on http://127.0.0.1:8080 ...");

    HttpServer::new(move || {
        App::new()
            .app_data(agent_data.clone())
            .service(get_q_table)
            .wrap(Cors::permissive())
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
