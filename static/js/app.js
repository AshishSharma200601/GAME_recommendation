/* Enhanced Game Recommender UI */
(function(){
  const resultsEl = document.getElementById('results');
  const emptyEl = document.getElementById('empty');
  const modal = document.getElementById('modal');
  const modalTitle = document.getElementById('modalTitle');
  const modalBody = document.getElementById('modalBody');
  const modalMeta = document.getElementById('modalMeta');
  const modalRating = document.getElementById('modalRating');
  const modalTags = document.getElementById('modalTags');
  const modalClose = document.getElementById('modalClose');
  const spinner = document.getElementById('spinner');
  const statsBar = document.getElementById('stats');
  
  let allResults = []; // Store all fetched results for client-side sorting
  let currentSort = 'relevance';

  function showSpinner(){ if(spinner) spinner.style.display='block'; }
  function hideSpinner(){ if(spinner) spinner.style.display='none'; }

  function fallbackImage(id){ return `https://picsum.photos/seed/${encodeURIComponent(id)}/800/1000`; }
  function tagPills(tagsStr){ 
    if(!tagsStr) return ''; 
    return tagsStr.split(/[;,]/).slice(0,6).map(t=>t.trim()).filter(Boolean).map(t=>`<span class="tag">${t}</span>`).join(''); 
  }

  function updateStats(results){
    if(!statsBar || !results || results.length === 0) {
      if(statsBar) statsBar.style.display = 'none';
      return;
    }
    
    statsBar.style.display = 'flex';
    const totalEl = document.getElementById('totalGames');
    const showingEl = document.getElementById('showingGames');
    const avgRatingEl = document.getElementById('avgRating');
    
    if(totalEl) totalEl.textContent = results.length;
    if(showingEl) showingEl.textContent = results.length;
    
    // Calculate average rating
    const ratings = results.map(r => parseFloat(r.rating || 0)).filter(r => r > 0);
    if(avgRatingEl) {
      if(ratings.length > 0) {
        const avg = (ratings.reduce((a,b) => a+b, 0) / ratings.length).toFixed(1);
        avgRatingEl.textContent = `${avg}/5.0 ⭐`;
      } else {
        avgRatingEl.textContent = '—';
      }
    }
  }

  function sortResults(results, sortBy){
    if(!results || results.length === 0) return results;
    
    const sorted = [...results];
    switch(sortBy){
      case 'rating':
        sorted.sort((a,b) => (parseFloat(b.rating) || 0) - (parseFloat(a.rating) || 0));
        break;
      case 'rating_asc':
        sorted.sort((a,b) => (parseFloat(a.rating) || 0) - (parseFloat(b.rating) || 0));
        break;
      case 'price':
        sorted.sort((a,b) => (parseFloat(a.price) || 0) - (parseFloat(b.price) || 0));
        break;
      case 'price_desc':
        sorted.sort((a,b) => (parseFloat(b.price) || 0) - (parseFloat(a.price) || 0));
        break;
      case 'year':
        sorted.sort((a,b) => (parseInt(b.release_year) || 0) - (parseInt(a.release_year) || 0));
        break;
      case 'year_asc':
        sorted.sort((a,b) => (parseInt(a.release_year) || 0) - (parseInt(b.release_year) || 0));
        break;
      case 'reviews':
        sorted.sort((a,b) => (parseInt(b.reviews_total) || 0) - (parseInt(a.reviews_total) || 0));
        break;
      case 'relevance':
      default:
        // Keep original order (sorted by relevance score from server)
        break;
    }
    return sorted;
  }

  function cardHTML(g){
    const id = g.game_id || g.title || Math.random();
    const bg = g.image_url && g.image_url.length ? g.image_url : fallbackImage(id);
    const desc = (g.description || '').trim();
    const descShort = desc.length > 120 ? desc.substring(0, 120) + '...' : desc;
    const priceStr = g.price && parseFloat(g.price) > 0 ? `$${parseFloat(g.price).toFixed(2)}` : 'Free';
    const yearStr = g.release_year && parseInt(g.release_year) > 0 ? g.release_year : 'N/A';
    const reviewsStr = g.reviews_total ? parseInt(g.reviews_total).toLocaleString() : '0';
    const rating = g.rating && parseFloat(g.rating) > 0 ? parseFloat(g.rating).toFixed(1) : null;
    const ratingStr = rating ? `⭐ ${rating}/5.0` : '';
    const genresStr = g.genres || 'N/A';
    const platformsStr = g.platforms || 'N/A';
    
    return `
      <div class="card" role="article" data-id="${id}" data-title="${(g.title||'').replace(/"/g,'') }" data-description="${(g.description||'').replace(/"/g,'') }" data-genres="${(g.genres||'')}" data-platforms="${(g.platforms||'')}" data-tags="${(g.tags||'')}" data-price="${g.price||''}" data-reviews="${g.reviews_total||0}" data-rating="${g.rating||''}" data-score="${Number(g.score).toFixed(4)}" data-year="${g.release_year||''}">
        <div class="bg" style="background-image:url('${bg.replace(/'/g,"\\'")}');"></div>
        ${rating ? `<div class="score rating-badge">${rating}/5.0 ⭐</div>` : `<div class="score">score: ${Number(g.score).toFixed(4)}</div>`}
        <div class="meta">
          <div class="title">${g.title || '—'}</div>
          ${descShort ? `<div class="description">${descShort}</div>` : ''}
          <div class="sub">
            <span class="detail-item">📅 ${yearStr}</span>
            <span class="detail-item">💰 ${priceStr}</span>
            ${ratingStr ? `<span class="detail-item">${ratingStr}</span>` : `<span class="detail-item">⭐ ${reviewsStr} reviews</span>`}
          </div>
          <div class="sub-secondary">
            <span class="detail-label">Genres:</span> ${genresStr} • 
            <span class="detail-label">Platforms:</span> ${platformsStr}
          </div>
          <div class="tags">${tagPills(g.tags)}</div>
        </div>
      </div>`;
  }

  function renderResults(results){
    if(!results || results.length === 0){
      resultsEl.innerHTML = '';
      emptyEl.style.display = 'block';
      statsBar.style.display = 'none';
      return;
    }
    
    emptyEl.style.display = 'none';
    const sorted = sortResults(results, currentSort);
    const html = sorted.map(cardHTML).join('');
    resultsEl.innerHTML = html;
    
    // Attach click handlers
    Array.from(resultsEl.querySelectorAll('.card')).forEach(c => {
      if(c._hasClick) return;
      c.addEventListener('click', (e)=>{
        const el = e.currentTarget;
        modalTitle.textContent = el.dataset.title || 'Details';
        
        const rating = el.dataset.rating && parseFloat(el.dataset.rating) > 0 ? parseFloat(el.dataset.rating).toFixed(1) : null;
        if(modalRating) {
          modalRating.textContent = rating ? `${rating}/5.0 ⭐` : '';
          modalRating.style.display = rating ? 'block' : 'none';
        }
        
        const desc = el.dataset.description || '';
        modalBody.innerHTML = desc ? 
          `<p style="line-height:1.7;color:var(--text);font-size:1rem">${desc}</p>` : 
          `<div style="color:var(--muted)">No description available</div>`;
        
        const yearStr = el.dataset.year && parseInt(el.dataset.year) > 0 ? el.dataset.year : 'N/A';
        const priceStr = el.dataset.price && parseFloat(el.dataset.price) > 0 ? `$${parseFloat(el.dataset.price).toFixed(2)}` : 'Free';
        const ratingStr = rating ? `${rating}/5.0 ⭐` : 'N/A';
        
        modalMeta.innerHTML = `
          <div><strong>Rating:</strong> ${ratingStr}</div>
          <div><strong>Release Year:</strong> ${yearStr}</div>
          <div><strong>Platforms:</strong> ${el.dataset.platforms || '—'}</div>
          <div><strong>Price:</strong> ${priceStr}</div>
          <div><strong>Reviews:</strong> ${parseInt(el.dataset.reviews || 0).toLocaleString()}</div>
          <div><strong>Genres:</strong> ${el.dataset.genres || '—'}</div>
        `;
        
        if(modalTags) {
          modalTags.innerHTML = tagPills(el.dataset.tags);
        }
        
        modal.style.display = 'flex';
        modal.setAttribute('aria-hidden','false');
      });
      c._hasClick = true;
    });
    
    updateStats(sorted);
  }

  // Pagination state
  let offset = 0;
  let lastTotal = null;

  async function fetchRecs({ append=false } = {}){
    showSpinner();
    
    const q = document.getElementById('q').value.trim();
    const genre = document.getElementById('genre').value.trim();
    const platform = document.getElementById('platform').value.trim();
    const max_price = document.getElementById('max_price').value.trim();
    const min_rating = document.getElementById('min_rating').value.trim();
    const k = parseInt(document.getElementById('k').value || 12, 10);
    
    const params = new URLSearchParams({ q, k, offset });
    if(genre) params.append('genre', genre);
    if(platform) params.append('platform', platform);
    if(max_price) params.append('max_price', max_price);

    try{
      const res = await fetch(`/recommend?${params.toString()}`, { cache:'no-store' });
      if(!res.ok){ 
        hideSpinner(); 
        alert('Failed to fetch: '+res.status); 
        return; 
      }
      
      const payload = await res.json();
      let data = payload && payload.results ? payload.results : [];
      const total = payload && payload.total ? payload.total : data.length;

      // Filter by min_rating on client side
      if(min_rating && parseFloat(min_rating) > 0){
        const minRatingVal = parseFloat(min_rating);
        data = data.filter(g => {
          const rating = parseFloat(g.rating || 0);
          return rating >= minRatingVal;
        });
      }

      if(!append){ 
        allResults = [];
        offset = 0; 
      }

      if(!Array.isArray(data) || data.length===0){
        if(!append) {
          renderResults([]);
        }
        hideSpinner();
        document.getElementById('loadMore').style.display = 'none';
        lastTotal = total;
        return;
      }

      // Deduplicate results by game_id or title
      function deduplicateResults(results) {
        const seen = new Set();
        return results.filter(game => {
          const identifier = (game.game_id || game.title || '').toString().toLowerCase().trim();
          if (!identifier || seen.has(identifier)) {
            return false;
          }
          seen.add(identifier);
          return true;
        });
      }

      if(append){
        allResults = deduplicateResults([...allResults, ...data]);
      } else {
        allResults = deduplicateResults(data);
      }

      renderResults(allResults);

      // Update offset and load-more visibility
      offset = offset + data.length;
      lastTotal = total;
      const loadMoreBtn = document.getElementById('loadMore');
      if(offset < total && allResults.length < total){ 
        loadMoreBtn.style.display = 'inline-block'; 
      } else { 
        loadMoreBtn.style.display = 'none'; 
      }

    }catch(err){ 
      console.error(err); 
      alert('Network error: '+err.message); 
    }
    hideSpinner();
  }

  function clearFilters(){
    document.getElementById('q').value = '';
    document.getElementById('genre').value = '';
    document.getElementById('platform').value = '';
    document.getElementById('max_price').value = '';
    document.getElementById('min_rating').value = '';
    document.getElementById('k').value = '12';
    document.getElementById('sortBy').value = 'relevance';
    currentSort = 'relevance';
    fetchRecs({ append:false });
  }


  // Debounce helper
  function debounce(fn, wait){ 
    let t; 
    return function(...args){ 
      clearTimeout(t); 
      t = setTimeout(()=>fn.apply(this,args), wait); 
    }; 
  }

  // Wire events
  document.getElementById('btn').addEventListener('click', ()=>fetchRecs({ append:false }));
  document.getElementById('clearBtn').addEventListener('click', clearFilters);
  document.getElementById('q').addEventListener('input', debounce(()=>fetchRecs({ append:false }), 700));
  document.getElementById('genre').addEventListener('input', debounce(()=>fetchRecs({ append:false }), 700));
  document.getElementById('platform').addEventListener('input', debounce(()=>fetchRecs({ append:false }), 700));
  document.getElementById('max_price').addEventListener('input', debounce(()=>fetchRecs({ append:false }), 700));
  document.getElementById('min_rating').addEventListener('input', debounce(()=>fetchRecs({ append:false }), 700));
  document.getElementById('k').addEventListener('change', ()=>fetchRecs({ append:false }));
  
  document.getElementById('sortBy').addEventListener('change', (e)=>{
    currentSort = e.target.value;
    renderResults(allResults);
  });
  
  document.getElementById('loadMore').addEventListener('click', ()=>fetchRecs({ append:true }));
  
  modalClose.addEventListener('click', ()=>{ 
    modal.style.display='none'; 
    modal.setAttribute('aria-hidden','true'); 
  });
  
  modal.addEventListener('click', (e)=>{
    if(e.target === modal || e.target.classList.contains('modal-backdrop')){
      modal.style.display='none';
      modal.setAttribute('aria-hidden','true');
    }
  });

  // Keyboard shortcuts
  document.addEventListener('keydown', (e)=>{
    if(e.key === 'Escape' && modal.style.display === 'flex'){
      modal.style.display='none';
      modal.setAttribute('aria-hidden','true');
    }
  });

  // Initial load
  window.addEventListener('load', ()=> setTimeout(()=>fetchRecs({ append:false }), 300));
})();
